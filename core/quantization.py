"""
Quantization overhead model.

Models the real costs of operating in reduced precision:
  - dequantization overhead (int8 -> fp32 on CPU has non-trivial cost)
  - memory layout penalties (non-contiguous access after quantization)
  - precision-loss induced recomputation (rare but real in some kernels)
  - dtype-specific roofline ceilings per hardware target

The key insight this module tries to capture: quantization is not free.
Below a certain arithmetic intensity threshold, the overhead of managing
quantized representations exceeds the memory bandwidth savings. That
threshold — the crossover point — is what we're trying to predict.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


class Dtype(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


# Bytes per element for each dtype
DTYPE_BYTES: dict[Dtype, float] = {
    Dtype.FP32: 4.0,
    Dtype.FP16: 2.0,
    Dtype.BF16: 2.0,
    Dtype.INT8: 1.0,
    Dtype.INT4: 0.5,
}

# Theoretical memory reduction vs fp32
MEMORY_REDUCTION: dict[Dtype, float] = {
    Dtype.FP32: 1.0,
    Dtype.FP16: 2.0,
    Dtype.BF16: 2.0,
    Dtype.INT8: 4.0,
    Dtype.INT4: 8.0,
}


@dataclass
class QuantizationConfig:
    dtype: Dtype

    # Fraction of ops that require dequantization back to fp32
    # (e.g. accumulation, normalization layers always need fp32)
    dequant_fraction: float = 0.0

    # Extra cycles per element for dequantization on this hardware
    dequant_cycles_per_element: float = 0.0

    # Memory layout penalty: quantized tensors often need repacking
    # expressed as a multiplier on memory access cycles (1.0 = no penalty)
    layout_penalty: float = 1.0

    # Scale/zero-point storage overhead as fraction of tensor size
    scale_overhead_fraction: float = 0.0

    # Whether this dtype requires separate accumulation buffers
    needs_accumulation_buffer: bool = False

    @property
    def effective_memory_reduction(self) -> float:
        """
        Actual memory reduction after accounting for scale storage
        and layout overhead.
        """
        base = MEMORY_REDUCTION[self.dtype]
        scale_cost = 1.0 + self.scale_overhead_fraction
        return base / (scale_cost * self.layout_penalty)

    @property
    def bytes_per_element(self) -> float:
        return DTYPE_BYTES[self.dtype]


@dataclass
class QuantizationOverheadResult:
    dtype: Dtype
    base_runtime_ms: float

    dequant_overhead_ms: float
    layout_overhead_ms: float
    scale_overhead_ms: float
    total_overhead_ms: float

    adjusted_runtime_ms: float
    effective_speedup_vs_fp32: float

    # The arithmetic intensity at which this dtype breaks even vs fp32
    crossover_ai: float

    # Whether this dtype is beneficial at the workload's actual AI
    is_beneficial: bool
    margin: float   # how far above/below the crossover point we are


@dataclass
class CrossoverAnalysis:
    """
    Full crossover point analysis across all dtypes for a given workload
    on a given hardware target.
    """
    hardware_name: str
    workload_label: str
    actual_ai: float

    fp32_runtime_ms: float
    results: list[QuantizationOverheadResult]

    def best_dtype(self) -> QuantizationOverheadResult:
        return min(self.results, key=lambda r: r.adjusted_runtime_ms)

    def crossover_summary(self) -> dict[str, float]:
        return {r.dtype.value: r.crossover_ai for r in self.results}

    def print(self):
        from rich.table import Table
        from rich.console import Console
        from rich import box

        t = Table(
            title=f"Quantization crossover — {self.workload_label} on {self.hardware_name}",
            box=box.SIMPLE_HEAD,
        )
        t.add_column("dtype",       style="bold")
        t.add_column("runtime ms",  justify="right")
        t.add_column("speedup",     justify="right")
        t.add_column("crossover AI",justify="right")
        t.add_column("beneficial",  justify="center")
        t.add_column("overhead ms", justify="right")

        for r in self.results:
            color = "green" if r.is_beneficial else "red"
            t.add_row(
                r.dtype.value,
                f"{r.adjusted_runtime_ms:.4f}",
                f"[{color}]{r.effective_speedup_vs_fp32:.2f}x[/{color}]",
                f"{r.crossover_ai:.2f}",
                "[green]yes[/green]" if r.is_beneficial else "[red]no[/red]",
                f"{r.total_overhead_ms:.4f}",
            )

        Console().print(t)


class QuantizationModel:
    """
    Models quantization overhead for a given hardware target.

    The core model:

      adjusted_runtime = base_runtime
                       + dequant_overhead
                       + layout_penalty_cost
                       + scale_storage_cost

    The crossover point is where adjusted_runtime(quantized) == runtime(fp32).
    Below this AI, quantization hurts. Above it, quantization helps.

    We derive crossover_ai analytically:

      Let B  = memory bandwidth (bytes/sec)
      Let F  = peak compute (flops/sec)
      Let r  = memory reduction ratio (e.g. 2x for fp16)
      Let d  = dequant overhead fraction
      Let l  = layout penalty multiplier

      fp32 runtime   = max(flops/F, bytes/B)
      quant runtime  = max(flops/F * compute_factor,
                           bytes/(r*B) * l) + d * flops/F

      crossover_ai   = F / (B * r) * (1 + d) / (1 - l/r)
                       (simplified, see _crossover_ai for full derivation)
    """

    def __init__(self, hardware_name: str, freq_ghz: float, peak_bandwidth_gb_s: float):
        self.hardware_name      = hardware_name
        self.freq_ghz           = freq_ghz
        self.peak_bw            = peak_bandwidth_gb_s * 1e9   # bytes/sec
        self._configs: dict[Dtype, QuantizationConfig] = {}

    def register_config(self, config: QuantizationConfig):
        self._configs[config.dtype] = config

    def get_config(self, dtype: Dtype) -> QuantizationConfig:
        return self._configs.get(dtype, _default_config(dtype))

    def overhead(
        self,
        dtype: Dtype,
        flops: int,
        bytes_fp32: int,
        base_runtime_ms: float,
        fp32_runtime_ms: float,
        actual_ai: float,
    ) -> QuantizationOverheadResult:
        cfg = self.get_config(dtype)

        if dtype == Dtype.FP32:
            return QuantizationOverheadResult(
                dtype=dtype,
                base_runtime_ms=base_runtime_ms,
                dequant_overhead_ms=0.0,
                layout_overhead_ms=0.0,
                scale_overhead_ms=0.0,
                total_overhead_ms=0.0,
                adjusted_runtime_ms=base_runtime_ms,
                effective_speedup_vs_fp32=1.0,
                crossover_ai=0.0,
                is_beneficial=True,
                margin=float("inf"),
            )

        freq = self.freq_ghz * 1e9

        # Dequantization overhead: fraction of elements need casting back.
        dequant_elements  = flops * cfg.dequant_fraction
        dequant_cycles    = dequant_elements * cfg.dequant_cycles_per_element
        dequant_ms        = (dequant_cycles / freq) * 1e3

        # Layout penalty: memory accesses take longer due to repacking.
        bytes_quant       = bytes_fp32 / MEMORY_REDUCTION[dtype]
        layout_extra_ms   = (base_runtime_ms
                             * (cfg.layout_penalty - 1.0)
                             * (bytes_quant / max(1, bytes_fp32)))

        # Scale/zero-point storage overhead.
        scale_bytes       = bytes_quant * cfg.scale_overhead_fraction
        scale_ms          = (scale_bytes / self.peak_bw) * 1e3

        total_overhead_ms  = dequant_ms + layout_extra_ms + scale_ms
        adjusted_ms        = base_runtime_ms + total_overhead_ms
        speedup            = fp32_runtime_ms / max(1e-12, adjusted_ms)

        crossover = self._crossover_ai(dtype, cfg)
        is_beneficial = actual_ai >= crossover
        margin = actual_ai - crossover

        return QuantizationOverheadResult(
            dtype=dtype,
            base_runtime_ms=base_runtime_ms,
            dequant_overhead_ms=dequant_ms,
            layout_overhead_ms=layout_extra_ms,
            scale_overhead_ms=scale_ms,
            total_overhead_ms=total_overhead_ms,
            adjusted_runtime_ms=adjusted_ms,
            effective_speedup_vs_fp32=speedup,
            crossover_ai=crossover,
            is_beneficial=is_beneficial,
            margin=margin,
        )

    def _crossover_ai(self, dtype: Dtype, cfg: QuantizationConfig) -> float:
        """
        Derive the arithmetic intensity crossover point analytically.

        At the crossover point, the quantized runtime equals the fp32 runtime.
        This happens when the memory savings are exactly cancelled by overhead.

        Simplified closed-form solution assuming memory-bound regime:

            crossover_ai = (overhead_fraction * peak_bw) / peak_compute_proxy

        We approximate peak_compute_proxy from the bandwidth and a typical
        ridge point for this hardware class.
        """
        r   = cfg.effective_memory_reduction
        d   = cfg.dequant_fraction * cfg.dequant_cycles_per_element
        l   = cfg.layout_penalty

        if r <= 1.0:
            return float("inf")

        # Memory savings factor: how much faster is memory access?
        mem_speedup = r / l

        # Overhead factor: how much does dequant cost relative to compute?
        overhead_factor = 1.0 + d * 0.01   # normalized

        # Crossover: below this AI, memory savings don't cover overhead
        if mem_speedup <= 1.0:
            return float("inf")

        crossover = (overhead_factor / (1.0 - 1.0 / mem_speedup)) * 2.0
        return max(0.0, crossover)

    def analyze_all_dtypes(
        self,
        workload_label: str,
        flops: int,
        bytes_fp32: int,
        fp32_runtime_ms: float,
        dtype_runtimes: dict[Dtype, float],
        actual_ai: float,
    ) -> CrossoverAnalysis:
        results = []
        for dtype, runtime_ms in dtype_runtimes.items():
            r = self.overhead(
                dtype=dtype,
                flops=flops,
                bytes_fp32=bytes_fp32,
                base_runtime_ms=runtime_ms,
                fp32_runtime_ms=fp32_runtime_ms,
                actual_ai=actual_ai,
            )
            results.append(r)

        return CrossoverAnalysis(
            hardware_name=self.hardware_name,
            workload_label=workload_label,
            actual_ai=actual_ai,
            fp32_runtime_ms=fp32_runtime_ms,
            results=results,
        )


# ---------------------------------------------------------------------------
# Default configs per dtype per hardware class
# ---------------------------------------------------------------------------

def _default_config(dtype: Dtype) -> QuantizationConfig:
    """Fallback config with conservative overhead estimates."""
    return QuantizationConfig(
        dtype=dtype,
        dequant_fraction=0.1,
        dequant_cycles_per_element=2.0,
        layout_penalty=1.05,
        scale_overhead_fraction=0.02,
        needs_accumulation_buffer=dtype in (Dtype.INT8, Dtype.INT4),
    )


def cpu_quantization_model(freq_ghz: float = 3.5) -> QuantizationModel:
    """
    CPU quantization overhead model.

    CPUs have significant dequantization costs because:
    - int8 VNNI instructions are available but require careful kernel design
    - Dequantization to fp32 for accumulation is common
    - Cache line alignment issues with sub-byte dtypes (int4)
    """
    model = QuantizationModel("CPU (x86 AVX-512)", freq_ghz, peak_bandwidth_gb_s=50.0)

    model.register_config(QuantizationConfig(
        dtype=Dtype.FP32,
        dequant_fraction=0.0,
        dequant_cycles_per_element=0.0,
        layout_penalty=1.0,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.FP16,
        dequant_fraction=0.15,      # some ops upcast to fp32
        dequant_cycles_per_element=1.5,
        layout_penalty=1.02,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.BF16,
        dequant_fraction=0.10,
        dequant_cycles_per_element=1.2,
        layout_penalty=1.01,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.INT8,
        dequant_fraction=0.30,      # accumulation always in int32, then cast
        dequant_cycles_per_element=3.0,
        layout_penalty=1.08,
        scale_overhead_fraction=0.03,
        needs_accumulation_buffer=True,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.INT4,
        dequant_fraction=0.50,      # very high dequant cost on CPU
        dequant_cycles_per_element=5.0,
        layout_penalty=1.20,        # significant repacking overhead
        scale_overhead_fraction=0.06,
        needs_accumulation_buffer=True,
    ))

    return model


def gpu_quantization_model(freq_ghz: float = 1.0) -> QuantizationModel:
    """
    GPU (T4-class) quantization overhead model.

    GPUs handle quantization much better than CPUs:
    - Tensor cores natively support fp16/bf16/int8
    - Dequantization is pipelined with memory loads
    - But int4 still has non-trivial software overhead on T4
    """
    model = QuantizationModel("GPU (T4)", freq_ghz, peak_bandwidth_gb_s=320.0)

    model.register_config(QuantizationConfig(
        dtype=Dtype.FP32,
        dequant_fraction=0.0,
        dequant_cycles_per_element=0.0,
        layout_penalty=1.0,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.FP16,
        dequant_fraction=0.05,      # tensor cores handle fp16 natively
        dequant_cycles_per_element=0.5,
        layout_penalty=1.0,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.BF16,
        dequant_fraction=0.05,
        dequant_cycles_per_element=0.5,
        layout_penalty=1.0,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.INT8,
        dequant_fraction=0.10,
        dequant_cycles_per_element=1.0,
        layout_penalty=1.03,
        scale_overhead_fraction=0.02,
        needs_accumulation_buffer=True,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.INT4,
        dequant_fraction=0.25,      # T4 has no native int4 tensor cores
        dequant_cycles_per_element=3.0,
        layout_penalty=1.15,
        scale_overhead_fraction=0.05,
        needs_accumulation_buffer=True,
    ))

    return model


def npu_quantization_model(freq_ghz: float = 1.25) -> QuantizationModel:
    """
    NPU/TPU quantization overhead model.

    NPUs are designed around quantized inference:
    - int8 is a first-class dtype, often faster than fp16
    - Dequantization is nearly free in the systolic array datapath
    - int4 support varies by generation
    """
    model = QuantizationModel("NPU (systolic + HBM)", freq_ghz, peak_bandwidth_gb_s=900.0)

    model.register_config(QuantizationConfig(
        dtype=Dtype.FP32,
        dequant_fraction=0.0,
        dequant_cycles_per_element=0.0,
        layout_penalty=1.0,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.FP16,
        dequant_fraction=0.02,
        dequant_cycles_per_element=0.3,
        layout_penalty=1.0,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.BF16,
        dequant_fraction=0.02,
        dequant_cycles_per_element=0.3,
        layout_penalty=1.0,
        scale_overhead_fraction=0.0,
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.INT8,
        dequant_fraction=0.05,      # systolic array pipelines this well
        dequant_cycles_per_element=0.5,
        layout_penalty=1.01,
        scale_overhead_fraction=0.01,
        needs_accumulation_buffer=False,  # built into the array
    ))
    model.register_config(QuantizationConfig(
        dtype=Dtype.INT4,
        dequant_fraction=0.15,
        dequant_cycles_per_element=1.5,
        layout_penalty=1.05,
        scale_overhead_fraction=0.03,
        needs_accumulation_buffer=True,
    ))

    return model
