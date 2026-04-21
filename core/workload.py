from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math

from .memory import AccessPattern
from .roofline import matmul_ai, conv2d_ai, attention_ai, elementwise_ai


class WorkloadKind(Enum):
    MATMUL      = "matmul"
    CONV2D      = "conv2d"
    ATTENTION   = "attention"
    ELEMENTWISE = "elementwise"
    LAYERNORM   = "layernorm"
    EMBEDDING   = "embedding"


DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 1,   # packed, approximation
}


@dataclass
class WorkloadSpec:
    kind: WorkloadKind
    label: str
    dtype: str = "fp32"
    # shape params vary by kind — stored as generic dict for flexibility
    params: dict = field(default_factory=dict)

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES.get(self.dtype, 4)

    def flop_count(self) -> int:
        raise NotImplementedError

    def bytes_accessed(self) -> int:
        raise NotImplementedError

    def arithmetic_intensity(self) -> float:
        return self.flop_count() / max(1, self.bytes_accessed())

    def memory_access_pattern(self) -> AccessPattern:
        raise NotImplementedError


@dataclass
class MatMulWorkload(WorkloadSpec):
    """C = A @ B,  A: (M, K),  B: (K, N),  C: (M, N)."""

    def __init__(self, M: int, N: int, K: int, dtype: str = "fp32", label: str = ""):
        super().__init__(
            kind=WorkloadKind.MATMUL,
            label=label or f"MatMul ({M}x{K}x{N})",
            dtype=dtype,
            params={"M": M, "N": N, "K": K},
        )

    @property
    def M(self): return self.params["M"]
    @property
    def N(self): return self.params["N"]
    @property
    def K(self): return self.params["K"]

    def flop_count(self) -> int:
        return 2 * self.M * self.N * self.K

    def bytes_accessed(self) -> int:
        db = self.dtype_bytes
        return db * (self.M * self.K + self.K * self.N + self.M * self.N)

    def arithmetic_intensity(self) -> float:
        return matmul_ai(self.M, self.N, self.K, self.dtype_bytes)

    def memory_access_pattern(self) -> AccessPattern:
        db = self.dtype_bytes
        working_set = self.bytes_accessed()
        return AccessPattern(
            working_set_bytes=working_set,
            stride_bytes=self.K * db,
            reuse_distance=self.K * db,
            sequential_fraction=0.75,
            read_write_ratio=0.85,
        )


@dataclass
class Conv2DWorkload(WorkloadSpec):
    """Standard 2D convolution."""

    def __init__(
        self,
        N: int, C_in: int, H: int, W: int,
        C_out: int, kH: int, kW: int,
        stride: int = 1, padding: int = 0,
        dtype: str = "fp32", label: str = "",
    ):
        super().__init__(
            kind=WorkloadKind.CONV2D,
            label=label or f"Conv2D ({N}x{C_in}x{H}x{W}, k={kH}x{kW}, out={C_out})",
            dtype=dtype,
            params=dict(N=N, C_in=C_in, H=H, W=W, C_out=C_out,
                        kH=kH, kW=kW, stride=stride, padding=padding),
        )

    @property
    def H_out(self):
        return (self.params["H"] + 2*self.params["padding"] - self.params["kH"]) // self.params["stride"] + 1
    @property
    def W_out(self):
        return (self.params["W"] + 2*self.params["padding"] - self.params["kW"]) // self.params["stride"] + 1

    def flop_count(self) -> int:
        p = self.params
        return (2 * p["N"] * p["C_out"] * self.H_out * self.W_out
                * p["C_in"] * p["kH"] * p["kW"])

    def bytes_accessed(self) -> int:
        p  = self.params
        db = self.dtype_bytes
        return db * (
            p["N"] * p["C_in"] * p["H"] * p["W"]
            + p["C_out"] * p["C_in"] * p["kH"] * p["kW"]
            + p["N"] * p["C_out"] * self.H_out * self.W_out
        )

    def arithmetic_intensity(self) -> float:
        p = self.params
        return conv2d_ai(
            p["N"], p["C_in"], p["H"], p["W"],
            p["C_out"], p["kH"], p["kW"], self.dtype_bytes,
        )

    def memory_access_pattern(self) -> AccessPattern:
        p  = self.params
        db = self.dtype_bytes
        return AccessPattern(
            working_set_bytes=self.bytes_accessed(),
            stride_bytes=p["W"] * p["C_in"] * db,
            reuse_distance=p["kH"] * p["W"] * p["C_in"] * db,
            sequential_fraction=0.6,
            read_write_ratio=0.88,
        )


@dataclass
class AttentionWorkload(WorkloadSpec):
    """Scaled dot-product attention."""

    def __init__(
        self,
        batch: int, heads: int, seq_len: int, head_dim: int,
        causal: bool = True,
        dtype: str = "fp32", label: str = "",
    ):
        super().__init__(
            kind=WorkloadKind.ATTENTION,
            label=label or f"Attention (B={batch}, H={heads}, S={seq_len}, D={head_dim})",
            dtype=dtype,
            params=dict(batch=batch, heads=heads, seq_len=seq_len,
                        head_dim=head_dim, causal=causal),
        )

    def flop_count(self) -> int:
        p = self.params
        B, H, S, D = p["batch"], p["heads"], p["seq_len"], p["head_dim"]
        qk_flops  = 2 * B * H * S * S * D
        av_flops  = 2 * B * H * S * S * D
        proj_flops = 2 * B * H * S * D * (H * D)
        causal_factor = 0.5 if p["causal"] else 1.0
        return int((qk_flops + av_flops) * causal_factor + proj_flops)

    def bytes_accessed(self) -> int:
        p  = self.params
        B, H, S, D = p["batch"], p["heads"], p["seq_len"], p["head_dim"]
        db = self.dtype_bytes
        qkv    = db * 3 * B * H * S * D
        attn   = db * B * H * S * S
        output = db * B * H * S * D
        return qkv + attn + output

    def arithmetic_intensity(self) -> float:
        p = self.params
        return attention_ai(
            p["batch"], p["heads"], p["seq_len"], p["head_dim"], self.dtype_bytes
        )

    def memory_access_pattern(self) -> AccessPattern:
        p  = self.params
        db = self.dtype_bytes
        return AccessPattern(
            working_set_bytes=self.bytes_accessed(),
            stride_bytes=p["head_dim"] * db,
            reuse_distance=p["seq_len"] * p["head_dim"] * db,
            sequential_fraction=0.5,
            read_write_ratio=0.80,
        )


@dataclass
class ElementwiseWorkload(WorkloadSpec):
    """Elementwise op: ReLU, GeLU, add, scale, etc."""

    def __init__(
        self,
        num_elements: int,
        op_name: str = "relu",
        dtype: str = "fp32",
        label: str = "",
    ):
        super().__init__(
            kind=WorkloadKind.ELEMENTWISE,
            label=label or f"Elementwise {op_name} ({num_elements:,} elements)",
            dtype=dtype,
            params=dict(num_elements=num_elements, op_name=op_name),
        )

    def flop_count(self) -> int:
        op = self.params["op_name"].lower()
        n  = self.params["num_elements"]
        # GeLU is more expensive than ReLU
        if op in ("gelu", "swish"):
            return n * 8
        return n

    def bytes_accessed(self) -> int:
        db = self.dtype_bytes
        return db * 2 * self.params["num_elements"]   # read + write

    def memory_access_pattern(self) -> AccessPattern:
        return AccessPattern(
            working_set_bytes=self.bytes_accessed(),
            stride_bytes=self.dtype_bytes,
            reuse_distance=0,
            sequential_fraction=1.0,
            read_write_ratio=0.5,
        )


# ---------------------------------------------------------------------------
# Pre-built workload suites
# ---------------------------------------------------------------------------

def llm_inference_suite(seq_len: int = 512, batch: int = 1) -> list:
    """Typical workloads from a single transformer decoder layer (GPT-like)."""
    d_model = 4096
    heads   = 32
    d_head  = d_model // heads

    return [
        MatMulWorkload(M=batch * seq_len, N=d_model, K=d_model,
                       dtype="fp16", label="QKV projection"),
        AttentionWorkload(batch=batch, heads=heads, seq_len=seq_len,
                          head_dim=d_head, causal=True,
                          dtype="fp16", label=f"Causal attention (S={seq_len})"),
        MatMulWorkload(M=batch * seq_len, N=d_model, K=d_model,
                       dtype="fp16", label="Output projection"),
        MatMulWorkload(M=batch * seq_len, N=d_model * 4, K=d_model,
                       dtype="fp16", label="FFN up-proj"),
        ElementwiseWorkload(num_elements=batch * seq_len * d_model * 4,
                            op_name="gelu", dtype="fp16",
                            label="FFN GeLU activation"),
        MatMulWorkload(M=batch * seq_len, N=d_model, K=d_model * 4,
                       dtype="fp16", label="FFN down-proj"),
    ]


def vision_model_suite() -> list:
    """ResNet-50-style conv stack (first few layers)."""
    return [
        Conv2DWorkload(N=1, C_in=3,   H=224, W=224, C_out=64,  kH=7, kW=7, stride=2, label="Conv1 7x7"),
        Conv2DWorkload(N=1, C_in=64,  H=56,  W=56,  C_out=64,  kH=3, kW=3, label="ResBlock 3x3 (a)"),
        Conv2DWorkload(N=1, C_in=64,  H=56,  W=56,  C_out=256, kH=1, kW=1, label="ResBlock 1x1 expand"),
        Conv2DWorkload(N=1, C_in=256, H=28,  W=28,  C_out=128, kH=3, kW=3, label="ResBlock 3x3 (b)"),
        ElementwiseWorkload(num_elements=256 * 56 * 56, op_name="relu", label="BN+ReLU"),
    ]


def matmul_scaling_suite() -> list:
    """Square matmul at different sizes to show roofline transition."""
    return [
        MatMulWorkload(M=32,   N=32,   K=32,   label="Tiny  (32³)"),
        MatMulWorkload(M=128,  N=128,  K=128,  label="Small (128³)"),
        MatMulWorkload(M=512,  N=512,  K=512,  label="Mid   (512³)"),
        MatMulWorkload(M=2048, N=2048, K=2048, label="Large (2048³)"),
        MatMulWorkload(M=8192, N=8192, K=8192, label="XL    (8192³)"),
    ]
