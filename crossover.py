"""
Crossover analysis engine.

Runs the full quantization crossover analysis:
  - Simulates each workload at every dtype on every hardware target
  - Computes the crossover AI for each dtype/hardware combination
  - Produces a calibration interface for fitting against real measurements
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from .sim import Simulator, HardwareConfig, WorkloadResult
from .core.quantization import (
    QuantizationModel, CrossoverAnalysis, Dtype,
    cpu_quantization_model, gpu_quantization_model, npu_quantization_model,
)
from .core.workload import WorkloadSpec
from .core.compute import ComputeArch


@dataclass
class HardwareQuantConfig:
    hw: HardwareConfig
    quant_model: QuantizationModel
    supported_dtypes: list[Dtype]


@dataclass
class CrossoverPoint:
    hardware_name: str
    dtype: Dtype
    crossover_ai: float
    # How confident we are in this prediction (0-1)
    # Will be updated after calibration against real measurements
    confidence: float = 0.5


@dataclass
class CalibrationPoint:
    """A real hardware measurement to calibrate the model against."""
    hardware_name: str
    workload_label: str
    dtype: Dtype
    measured_runtime_ms: float
    matrix_size: int    # M=N=K for square matmuls
    arithmetic_intensity: float


@dataclass
class CalibrationResult:
    hardware_name: str
    dtype: Dtype
    predicted_ms: float
    measured_ms: float
    error_pct: float
    within_threshold: bool   # within 20% = good calibration


class CrossoverEngine:
    def __init__(self, configs: list[HardwareQuantConfig]):
        self.configs = {c.hw.name: c for c in configs}

    def run_workload_all_dtypes(
        self,
        workload: WorkloadSpec,
        hw_name: str,
    ) -> CrossoverAnalysis:
        cfg = self.configs[hw_name]
        sim = Simulator(cfg.hw)

        fp32_result = sim.run_workload(workload, dtype_override="fp32")
        fp32_ms     = fp32_result.runtime_ms

        dtype_runtimes: dict[Dtype, float] = {Dtype.FP32: fp32_ms}

        for dtype in cfg.supported_dtypes:
            if dtype == Dtype.FP32:
                continue
            result = sim.run_workload(workload, dtype_override=dtype.value)
            dtype_runtimes[dtype] = result.runtime_ms

        return cfg.quant_model.analyze_all_dtypes(
            workload_label=workload.label,
            flops=workload.flop_count(),
            bytes_fp32=workload.bytes_accessed(),
            fp32_runtime_ms=fp32_ms,
            dtype_runtimes=dtype_runtimes,
            actual_ai=workload.arithmetic_intensity(),
        )

    def sweep_matrix_sizes(
        self,
        sizes: list[int],
        hw_name: str,
        dtypes: Optional[list[Dtype]] = None,
    ) -> list[CrossoverAnalysis]:
        """
        Sweep square matmul sizes to find where crossover points shift.
        This is the core experiment that matches the Colab benchmark.
        """
        from .core.workload import MatMulWorkload

        cfg    = self.configs[hw_name]
        dtypes = dtypes or cfg.supported_dtypes
        results = []

        for size in sizes:
            wl = MatMulWorkload(M=size, N=size, K=size,
                                label=f"MatMul {size}x{size}x{size}")
            analysis = self.run_workload_all_dtypes(wl, hw_name)
            results.append(analysis)

        return results

    def predict_crossover_points(self, hw_name: str) -> list[CrossoverPoint]:
        """
        Predict crossover AI for each dtype on the given hardware.
        Uses the quantization model's analytical formula.
        """
        cfg    = self.configs[hw_name]
        points = []

        for dtype in cfg.supported_dtypes:
            if dtype == Dtype.FP32:
                continue
            quant_cfg = cfg.quant_model.get_config(dtype)
            crossover = cfg.quant_model._crossover_ai(dtype, quant_cfg)
            points.append(CrossoverPoint(
                hardware_name=hw_name,
                dtype=dtype,
                crossover_ai=crossover,
            ))

        return points

    def calibrate(
        self,
        measurements: list[CalibrationPoint],
    ) -> list[CalibrationResult]:
        """
        Compare simulator predictions against real measurements.
        Updates confidence scores on crossover points.
        """
        from .core.workload import MatMulWorkload

        results = []
        for m in measurements:
            if m.hardware_name not in self.configs:
                continue

            cfg = self.configs[m.hardware_name]
            sim = Simulator(cfg.hw)
            wl  = MatMulWorkload(
                M=m.matrix_size, N=m.matrix_size, K=m.matrix_size,
                dtype=m.dtype.value,
                label=m.workload_label,
            )
            predicted = sim.run_workload(wl, dtype_override=m.dtype.value)
            pred_ms   = predicted.runtime_ms
            error_pct = abs(pred_ms - m.measured_runtime_ms) / max(1e-9, m.measured_runtime_ms) * 100

            results.append(CalibrationResult(
                hardware_name=m.hardware_name,
                dtype=m.dtype,
                predicted_ms=pred_ms,
                measured_ms=m.measured_runtime_ms,
                error_pct=error_pct,
                within_threshold=error_pct <= 20.0,
            ))

        return results

    def print_calibration(self, results: list[CalibrationResult]):
        from rich.table import Table
        from rich.console import Console
        from rich import box

        t = Table(title="Calibration: simulated vs measured", box=box.SIMPLE_HEAD)
        t.add_column("Hardware",   style="bold", min_width=20)
        t.add_column("dtype",      min_width=6)
        t.add_column("Predicted",  justify="right")
        t.add_column("Measured",   justify="right")
        t.add_column("Error",      justify="right")
        t.add_column("Status",     justify="center")

        for r in results:
            color  = "green" if r.within_threshold else "red"
            status = "[green]OK[/green]" if r.within_threshold else "[red]MISS[/red]"
            t.add_row(
                r.hardware_name,
                r.dtype.value,
                f"{r.predicted_ms:.4f} ms",
                f"{r.measured_ms:.4f} ms",
                f"[{color}]{r.error_pct:.1f}%[/{color}]",
                status,
            )

        Console().print(t)

    def print_crossover_summary(self, hw_name: str):
        from rich.table import Table
        from rich.console import Console
        from rich import box

        points = self.predict_crossover_points(hw_name)

        t = Table(
            title=f"Predicted crossover points — {hw_name}",
            box=box.SIMPLE_HEAD,
        )
        t.add_column("dtype",        style="bold")
        t.add_column("crossover AI", justify="right")
        t.add_column("interpretation")

        for p in points:
            ai = p.crossover_ai
            if ai < 1:
                interp = "almost always beneficial"
            elif ai < 5:
                interp = "beneficial for most ML ops"
            elif ai < 20:
                interp = "beneficial only for large matmuls"
            elif ai < 100:
                interp = "marginal — only very compute-heavy ops"
            else:
                interp = "rarely beneficial on this hardware"

            t.add_row(p.dtype.value, f"{ai:.2f}", interp)

        Console().print(t)


# ---------------------------------------------------------------------------
# Pre-built engine configs
# ---------------------------------------------------------------------------

def default_cpu_config() -> HardwareQuantConfig:
    from .core import cpu_desktop_hierarchy, cpu_avx512
    from .sim import HardwareConfig

    return HardwareQuantConfig(
        hw=HardwareConfig(
            name="CPU (x86 AVX-512)",
            memory=cpu_desktop_hierarchy(),
            compute=cpu_avx512(),
        ),
        quant_model=cpu_quantization_model(),
        supported_dtypes=[Dtype.FP32, Dtype.FP16, Dtype.BF16, Dtype.INT8, Dtype.INT4],
    )


def default_gpu_config() -> HardwareQuantConfig:
    """T4-class GPU config for Colab free tier."""
    from .core import MemoryHierarchy, CacheConfig, MemLevel
    from .core.compute import ComputeUnitConfig, ComputeArch
    from .sim import HardwareConfig

    freq = 1.0
    mem = MemoryHierarchy(
        name="GPU T4",
        freq_ghz=freq,
        levels={
            MemLevel.L1: CacheConfig(
                size_kb=64, line_bytes=128, associativity=4,
                hit_latency_cycles=28, bandwidth_gb_s=6000,
                miss_penalty_cycles=100,
            ),
            MemLevel.L2: CacheConfig(
                size_kb=4096, line_bytes=128, associativity=8,
                hit_latency_cycles=193, bandwidth_gb_s=1600,
                miss_penalty_cycles=300,
            ),
            MemLevel.DRAM: CacheConfig(
                size_kb=16 * 1024 * 1024, line_bytes=128, associativity=1,
                hit_latency_cycles=400, bandwidth_gb_s=320,
                miss_penalty_cycles=0,
            ),
        },
    )

    # T4: 8.1 TFLOPS fp32, 65 TOPS int8, 130 TOPS int4
    compute = ComputeUnitConfig(
        arch=ComputeArch.SYSTOLIC,
        freq_ghz=freq,
        num_cores=2560,
        simd_width_floats=8,
        peak_tflops=8.1,
        mac_efficiency=0.528,
        memory_overlap_factor=0.92,  # T4 hides ~92% of memory latency via warp switching
        dtype_speedup={
            "fp32": 1.0, "fp16": 5.55, "bf16": 0.61,
            "int8": 4.21, "int4": 8.0,
        },
    )

    return HardwareQuantConfig(
        hw=HardwareConfig(name="GPU (T4)", memory=mem, compute=compute),
        quant_model=gpu_quantization_model(),
        supported_dtypes=[Dtype.FP32, Dtype.FP16, Dtype.BF16, Dtype.INT8],
    )


def default_npu_config() -> HardwareQuantConfig:
    from .core import npu_hierarchy, npu_systolic
    from .sim import HardwareConfig

    return HardwareQuantConfig(
        hw=HardwareConfig(
            name="NPU (systolic + HBM)",
            memory=npu_hierarchy(),
            compute=npu_systolic(),
        ),
        quant_model=npu_quantization_model(),
        supported_dtypes=[Dtype.FP32, Dtype.FP16, Dtype.BF16, Dtype.INT8, Dtype.INT4],
    )


def build_default_engine() -> CrossoverEngine:
    return CrossoverEngine([
        default_cpu_config(),
        default_gpu_config(),
        default_npu_config(),
    ])
