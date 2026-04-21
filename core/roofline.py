from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class RooflinePoint:
    label: str
    arithmetic_intensity: float   # FLOPs per byte
    achieved_flops_per_sec: float
    peak_flops_per_sec: float
    peak_bandwidth_bytes_per_sec: float
    is_compute_bound: bool

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at which compute and memory limits intersect."""
        return self.peak_flops_per_sec / self.peak_bandwidth_bytes_per_sec

    @property
    def roofline_ceiling(self) -> float:
        """Theoretical max performance at this arithmetic intensity."""
        return min(
            self.peak_flops_per_sec,
            self.arithmetic_intensity * self.peak_bandwidth_bytes_per_sec,
        )

    @property
    def efficiency(self) -> float:
        """Fraction of roofline ceiling achieved."""
        ceiling = self.roofline_ceiling
        if ceiling <= 0:
            return 0.0
        return min(1.0, self.achieved_flops_per_sec / ceiling)

    @property
    def bottleneck(self) -> str:
        return "compute-bound" if self.is_compute_bound else "memory-bound"

    @property
    def gap_to_ceiling_x(self) -> float:
        """How many times faster could we go if we closed the gap?"""
        ceiling = self.roofline_ceiling
        achieved = max(1.0, self.achieved_flops_per_sec)
        return ceiling / achieved


class RooflineModel:
    def __init__(
        self,
        peak_flops_per_sec: float,
        peak_bandwidth_bytes_per_sec: float,
        label: str = "hardware",
    ):
        self.peak_flops      = peak_flops_per_sec
        self.peak_bw         = peak_bandwidth_bytes_per_sec
        self.label           = label

    @property
    def ridge_point(self) -> float:
        return self.peak_flops / self.peak_bw

    def evaluate(
        self,
        flops: int,
        bytes_accessed: int,
        runtime_s: float,
        workload_label: str = "op",
    ) -> RooflinePoint:
        ai = flops / max(1, bytes_accessed)
        achieved = flops / max(1e-15, runtime_s)
        compute_limited = ai >= self.ridge_point

        return RooflinePoint(
            label=workload_label,
            arithmetic_intensity=ai,
            achieved_flops_per_sec=achieved,
            peak_flops_per_sec=self.peak_flops,
            peak_bandwidth_bytes_per_sec=self.peak_bw,
            is_compute_bound=compute_limited,
        )

    def theoretical_peak_runtime_s(self, flops: int, bytes_accessed: int) -> float:
        ai = flops / max(1, bytes_accessed)
        peak_perf = min(self.peak_flops, ai * self.peak_bw)
        return flops / max(1.0, peak_perf)


def build_roofline(
    peak_tflops: float,
    peak_bandwidth_gb_s: float,
    label: str,
) -> RooflineModel:
    return RooflineModel(
        peak_flops_per_sec=peak_tflops * 1e12,
        peak_bandwidth_bytes_per_sec=peak_bandwidth_gb_s * 1e9,
        label=label,
    )


# ---------------------------------------------------------------------------
# Arithmetic intensity helpers for common ML ops
# ---------------------------------------------------------------------------

def matmul_ai(M: int, N: int, K: int, dtype_bytes: int = 4) -> float:
    """
    For C = A @ B, A is (M,K), B is (K,N), C is (M,N).
    FLOPs = 2*M*N*K (multiply-accumulate).
    Bytes = dtype_bytes * (M*K + K*N + M*N).
    """
    flops = 2 * M * N * K
    bytes_accessed = dtype_bytes * (M * K + K * N + M * N)
    return flops / max(1, bytes_accessed)


def conv2d_ai(
    N: int, C_in: int, H: int, W: int,
    C_out: int, kH: int, kW: int,
    dtype_bytes: int = 4,
) -> float:
    H_out = H - kH + 1
    W_out = W - kW + 1
    flops = 2 * N * C_out * H_out * W_out * C_in * kH * kW
    bytes_input   = dtype_bytes * N * C_in * H * W
    bytes_weights = dtype_bytes * C_out * C_in * kH * kW
    bytes_output  = dtype_bytes * N * C_out * H_out * W_out
    return flops / max(1, bytes_input + bytes_weights + bytes_output)


def attention_ai(
    batch: int, heads: int, seq_len: int, head_dim: int,
    dtype_bytes: int = 4,
) -> float:
    """
    Scaled dot-product attention.
    QK^T: (B,H,S,D) x (B,H,D,S) → (B,H,S,S)   flops = 2*B*H*S*S*D
    softmax + AV:                                flops ≈ 2*B*H*S*S*D again
    Total bytes = Q + K + V + output tensors.
    """
    B, H, S, D = batch, heads, seq_len, head_dim
    flops = 4 * B * H * S * S * D
    bytes_qkv    = dtype_bytes * 3 * B * H * S * D
    bytes_attn   = dtype_bytes * B * H * S * S
    bytes_output = dtype_bytes * B * H * S * D
    return flops / max(1, bytes_qkv + bytes_attn + bytes_output)


def elementwise_ai(num_elements: int, dtype_bytes: int = 4) -> float:
    """Elementwise ops (ReLU, add, etc.) — 1 FLOP per element, read+write."""
    return 1.0 / (2 * dtype_bytes)  # bandwidth-bound by definition
