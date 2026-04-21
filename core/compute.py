from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


class ComputeArch(Enum):
    CPU_SCALAR   = "CPU scalar"
    CPU_SIMD     = "CPU SIMD (AVX-512)"
    SYSTOLIC     = "Systolic array (NPU)"
    MOBILE_NEON  = "Mobile NEON (ARM)"


@dataclass
class ComputeUnitConfig:
    arch: ComputeArch
    freq_ghz: float
    num_cores: int            # parallel execution units / PEs
    simd_width_floats: int    # FP32 ops per cycle per core
    peak_tflops: float        # theoretical peak (FP32)
    mac_efficiency: float     # fraction of peak achievable in practice
    dtype_speedup: dict       # relative throughput for fp16, int8, etc.

    @property
    def peak_gflops(self) -> float:
        return self.peak_tflops * 1000

    def effective_gflops(self, dtype: str = "fp32") -> float:
        speedup = self.dtype_speedup.get(dtype, 1.0)
        return self.peak_gflops * self.mac_efficiency * speedup


@dataclass
class ComputeResult:
    arch: ComputeArch
    op_count: int             # total FLOPs
    cycles: float
    runtime_ms: float
    utilization: float
    effective_tflops: float
    dtype: str
    bottleneck: str           # "compute" or "memory"


class ComputeSimulator:
    def __init__(self, config: ComputeUnitConfig):
        self.cfg = config

    def simulate(
        self,
        flops: int,
        memory_stall_cycles: float,
        dtype: str = "fp32",
        tiling_efficiency: float = 0.85,
    ) -> ComputeResult:
        freq = self.cfg.freq_ghz * 1e9

        # How many cycles does the compute itself take?
        effective_gflops = self.cfg.effective_gflops(dtype) * tiling_efficiency
        compute_cycles   = (flops / (effective_gflops * 1e9)) * freq

        # Total cycles is max of compute and memory (overlap model).
        # Stalls that exceed compute time add directly to total.
        overlap_factor = 0.6   # fraction of memory latency hidden by out-of-order / prefetch
        exposed_stall  = memory_stall_cycles * (1.0 - overlap_factor)
        total_cycles   = compute_cycles + exposed_stall

        runtime_ms      = (total_cycles / freq) * 1e3
        utilization     = compute_cycles / max(1.0, total_cycles)
        effective_tflops = (flops / max(1e-12, runtime_ms * 1e-3)) / 1e12

        bottleneck = "memory" if exposed_stall > compute_cycles * 0.2 else "compute"

        return ComputeResult(
            arch=self.cfg.arch,
            op_count=flops,
            cycles=total_cycles,
            runtime_ms=runtime_ms,
            utilization=utilization,
            effective_tflops=effective_tflops,
            dtype=dtype,
            bottleneck=bottleneck,
        )

    def systolic_array_efficiency(self, M: int, N: int, K: int) -> float:
        """
        Systolic arrays hit peak efficiency only when tiles map cleanly
        onto the array dimensions. Simulate the utilization falloff for
        non-square or oddly-shaped matmuls.
        """
        if self.cfg.arch != ComputeArch.SYSTOLIC:
            return self.cfg.mac_efficiency

        array_dim = int(math.sqrt(self.cfg.num_cores))
        m_eff = min(1.0, M / array_dim)
        n_eff = min(1.0, N / array_dim)
        k_eff = min(1.0, K / 16)   # pipeline depth warmup
        return self.cfg.mac_efficiency * m_eff * n_eff * k_eff


# ---------------------------------------------------------------------------
# Pre-built configs
# ---------------------------------------------------------------------------

def cpu_avx512(freq_ghz: float = 3.5) -> ComputeUnitConfig:
    """
    Approximates a modern desktop CPU with AVX-512 (e.g. Intel Xeon / Core i9).
    16x FP32 per cycle per core, 8 cores.
    """
    cores = 8
    simd  = 16  # AVX-512: 512-bit / 32-bit = 16 floats
    peak  = freq_ghz * cores * simd * 2 / 1000  # *2 for FMA
    return ComputeUnitConfig(
        arch=ComputeArch.CPU_SIMD,
        freq_ghz=freq_ghz,
        num_cores=cores,
        simd_width_floats=simd,
        peak_tflops=peak,
        mac_efficiency=0.55,
        dtype_speedup={"fp32": 1.0, "fp16": 1.8, "int8": 3.0, "bf16": 1.5},
    )


def npu_systolic(freq_ghz: float = 1.25) -> ComputeUnitConfig:
    """
    Approximates a 256x256 systolic array NPU (roughly TPU v3 class).
    """
    array_size = 256 * 256
    peak = freq_ghz * array_size * 2 / 1e3   # MACs * 2 FLOPs/MAC / 1e3 to TFLOPS
    return ComputeUnitConfig(
        arch=ComputeArch.SYSTOLIC,
        freq_ghz=freq_ghz,
        num_cores=array_size,
        simd_width_floats=256,
        peak_tflops=peak,
        mac_efficiency=0.80,
        dtype_speedup={"fp32": 1.0, "fp16": 2.0, "int8": 4.0, "bf16": 2.0},
    )


def mobile_arm_neon(freq_ghz: float = 2.4) -> ComputeUnitConfig:
    """
    Approximates an ARM Cortex-A78 / Apple Firestorm class core.
    4 FP32 per cycle per core, 4 big cores.
    """
    cores = 4
    simd  = 4
    peak  = freq_ghz * cores * simd * 2 / 1000
    return ComputeUnitConfig(
        arch=ComputeArch.MOBILE_NEON,
        freq_ghz=freq_ghz,
        num_cores=cores,
        simd_width_floats=simd,
        peak_tflops=peak,
        mac_efficiency=0.50,
        dtype_speedup={"fp32": 1.0, "fp16": 2.0, "int8": 4.0, "bf16": 1.5},
    )
