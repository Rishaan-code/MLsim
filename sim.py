from dataclasses import dataclass, field
from typing import Optional
import math

from .core.memory  import MemoryHierarchy, MemorySimulator, MemLevel
from .core.compute import ComputeUnitConfig, ComputeSimulator, ComputeArch
from .core.roofline import RooflineModel, RooflinePoint, build_roofline
from .core.workload import WorkloadSpec, WorkloadKind


@dataclass
class HardwareConfig:
    name: str
    memory:  MemoryHierarchy
    compute: ComputeUnitConfig

    def roofline_model(self, dtype: str = "fp32") -> RooflineModel:
        peak_tflops = self.compute.effective_gflops(dtype) / 1000
        # Use the highest-bandwidth memory level as the bandwidth ceiling.
        bw_levels = sorted(
            self.memory.levels.items(),
            key=lambda kv: kv[1].bandwidth_gb_s,
            reverse=True,
        )
        peak_bw = bw_levels[0][1].bandwidth_gb_s
        return build_roofline(peak_tflops, peak_bw, label=self.name)


@dataclass
class WorkloadResult:
    workload_label: str
    workload_kind:  WorkloadKind
    hardware_name:  str
    dtype:          str

    flops:          int
    bytes_accessed: int
    arithmetic_intensity: float

    runtime_ms:         float
    compute_cycles:     float
    memory_stall_cycles: float

    compute_utilization: float
    effective_tflops:    float
    bottleneck:          str

    cache_hit_rate:      float
    bw_utilization:      float

    roofline: RooflinePoint


@dataclass
class SimulationRun:
    hardware: HardwareConfig
    results:  list[WorkloadResult] = field(default_factory=list)

    def add(self, r: WorkloadResult):
        self.results.append(r)

    def total_runtime_ms(self) -> float:
        return sum(r.runtime_ms for r in self.results)

    def by_bottleneck(self) -> dict[str, list[WorkloadResult]]:
        out: dict[str, list] = {"compute": [], "memory": []}
        for r in self.results:
            out[r.bottleneck].append(r)
        return out

    def average_utilization(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.compute_utilization for r in self.results) / len(self.results)


class Simulator:
    def __init__(self, hardware: HardwareConfig):
        self.hw       = hardware
        self.mem_sim  = MemorySimulator(hardware.memory)
        self.cmp_sim  = ComputeSimulator(hardware.compute)

    def run_workload(
        self,
        workload: WorkloadSpec,
        dtype_override: Optional[str] = None,
        tiling_efficiency: float = 0.85,
    ) -> WorkloadResult:
        dtype   = dtype_override or workload.dtype
        pattern = workload.memory_access_pattern()
        flops   = workload.flop_count()
        nbytes  = workload.bytes_accessed()

        # Simulate memory subsystem.
        self.mem_sim.reset()
        mem_result = self.mem_sim.simulate_access(nbytes, pattern)

        # Tiling / systolic efficiency adjustment.
        if self.hw.compute.arch == ComputeArch.SYSTOLIC:
            eff = self._systolic_tile_efficiency(workload)
        else:
            eff = tiling_efficiency

        # Simulate compute.
        cmp_result = self.cmp_sim.simulate(
            flops=flops,
            memory_stall_cycles=mem_result.stall_cycles,
            dtype=dtype,
            tiling_efficiency=eff,
        )

        # Roofline evaluation.
        rf_model = self.hw.roofline_model(dtype)
        runtime_s = cmp_result.runtime_ms * 1e-3
        rf_point  = rf_model.evaluate(flops, nbytes, runtime_s, workload.label)

        return WorkloadResult(
            workload_label=workload.label,
            workload_kind=workload.kind,
            hardware_name=self.hw.name,
            dtype=dtype,
            flops=flops,
            bytes_accessed=nbytes,
            arithmetic_intensity=workload.arithmetic_intensity(),
            runtime_ms=cmp_result.runtime_ms,
            compute_cycles=cmp_result.cycles,
            memory_stall_cycles=mem_result.stall_cycles,
            compute_utilization=cmp_result.utilization,
            effective_tflops=cmp_result.effective_tflops,
            bottleneck=cmp_result.bottleneck,
            cache_hit_rate=mem_result.hit_rate,
            bw_utilization=mem_result.bandwidth_utilization,
            roofline=rf_point,
        )

    def run_suite(
        self,
        workloads: list[WorkloadSpec],
        dtype_override: Optional[str] = None,
    ) -> SimulationRun:
        run = SimulationRun(hardware=self.hw)
        for wl in workloads:
            result = self.run_workload(wl, dtype_override=dtype_override)
            run.add(result)
        return run

    def _systolic_tile_efficiency(self, workload: WorkloadSpec) -> float:
        from .core.workload import MatMulWorkload, AttentionWorkload
        if isinstance(workload, MatMulWorkload):
            return self.cmp_sim.systolic_array_efficiency(
                workload.M, workload.N, workload.K
            )
        if isinstance(workload, AttentionWorkload):
            p = workload.params
            S, D = p["seq_len"], p["head_dim"]
            return self.cmp_sim.systolic_array_efficiency(S, S, D)
        return 0.80


def compare_hardware(
    workloads: list[WorkloadSpec],
    hardware_configs: list[HardwareConfig],
    dtype_override: Optional[str] = None,
) -> dict[str, SimulationRun]:
    results = {}
    for hw in hardware_configs:
        sim = Simulator(hw)
        results[hw.name] = sim.run_suite(workloads, dtype_override=dtype_override)
    return results
