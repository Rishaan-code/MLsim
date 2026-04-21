from dataclasses import dataclass
from typing import Optional
import math

from .sim import SimulationRun, WorkloadResult


@dataclass
class SpeedupTable:
    baseline_name: str
    target_name:   str
    rows: list[dict]   # per-workload speedup breakdown

    def print(self):
        from rich.table import Table
        from rich.console import Console
        from rich import box

        t = Table(
            title=f"Speedup: {self.target_name} vs {self.baseline_name}",
            box=box.SIMPLE_HEAD,
            show_lines=False,
        )
        t.add_column("Workload",   style="bold", min_width=30)
        t.add_column("Speedup",    justify="right")
        t.add_column("Bottleneck", justify="center")
        t.add_column("AI",         justify="right")
        t.add_column("Base ms",    justify="right")
        t.add_column("Target ms",  justify="right")

        for row in self.rows:
            speedup = row["speedup"]
            color   = "green" if speedup >= 2 else ("yellow" if speedup >= 1 else "red")
            t.add_row(
                row["label"],
                f"[{color}]{speedup:.2f}x[/{color}]",
                row["bottleneck"],
                f"{row['arithmetic_intensity']:.2f}",
                f"{row['baseline_ms']:.3f}",
                f"{row['target_ms']:.3f}",
            )

        Console().print(t)


@dataclass
class MemoryPressureReport:
    hardware_name: str
    workload_label: str
    working_set_mb: float
    cache_hit_rate: float
    stall_cycles: float
    dominant_level: str
    pressure: str   # "low" / "medium" / "high"

    def __str__(self):
        return (
            f"{self.workload_label} on {self.hardware_name}: "
            f"WS={self.working_set_mb:.2f}MB  "
            f"hit={self.cache_hit_rate:.1%}  "
            f"stalls={self.stall_cycles:.0f}  "
            f"pressure={self.pressure}"
        )


def compute_speedup_table(
    baseline: SimulationRun,
    target: SimulationRun,
) -> SpeedupTable:
    baseline_map = {r.workload_label: r for r in baseline.results}
    rows = []
    for r in target.results:
        b = baseline_map.get(r.workload_label)
        if b is None:
            continue
        speedup = b.runtime_ms / max(1e-9, r.runtime_ms)
        rows.append({
            "label":                r.workload_label,
            "speedup":              speedup,
            "bottleneck":           r.bottleneck,
            "arithmetic_intensity": r.arithmetic_intensity,
            "baseline_ms":          b.runtime_ms,
            "target_ms":            r.runtime_ms,
        })
    return SpeedupTable(
        baseline_name=baseline.hardware.name,
        target_name=target.hardware.name,
        rows=rows,
    )


def memory_pressure_report(run: SimulationRun) -> list[MemoryPressureReport]:
    reports = []
    for r in run.results:
        ws_mb = r.bytes_accessed / (1024 ** 2)
        hit   = r.cache_hit_rate

        if hit >= 0.90:
            pressure = "low"
        elif hit >= 0.70:
            pressure = "medium"
        else:
            pressure = "high"

        reports.append(MemoryPressureReport(
            hardware_name=r.hardware_name,
            workload_label=r.workload_label,
            working_set_mb=ws_mb,
            cache_hit_rate=hit,
            stall_cycles=r.memory_stall_cycles,
            dominant_level="L1" if hit > 0.95 else ("L2" if hit > 0.80 else "DRAM"),
            pressure=pressure,
        ))
    return reports


def suite_summary(run: SimulationRun) -> dict:
    results = run.results
    if not results:
        return {}

    total_flops   = sum(r.flops          for r in results)
    total_bytes   = sum(r.bytes_accessed for r in results)
    total_runtime = run.total_runtime_ms()
    avg_util      = run.average_utilization()
    avg_hit       = sum(r.cache_hit_rate for r in results) / len(results)

    mem_bound     = sum(1 for r in results if r.bottleneck == "memory")
    compute_bound = len(results) - mem_bound

    return {
        "hardware":            run.hardware.name,
        "workload_count":      len(results),
        "total_flops":         total_flops,
        "total_bytes_accessed":total_bytes,
        "overall_ai":          total_flops / max(1, total_bytes),
        "total_runtime_ms":    total_runtime,
        "avg_compute_util":    avg_util,
        "avg_cache_hit_rate":  avg_hit,
        "memory_bound_ops":    mem_bound,
        "compute_bound_ops":   compute_bound,
        "peak_tflops_hw":      run.hardware.compute.effective_gflops() / 1000,
        "achieved_tflops":     (total_flops / max(1e-9, total_runtime * 1e-3)) / 1e12,
    }


def print_suite_summary(run: SimulationRun):
    from rich.console import Console
    from rich.table   import Table
    from rich         import box

    s = suite_summary(run)
    console = Console()

    console.rule(f"[bold]{s['hardware']}[/bold]")

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("metric", style="dim", min_width=28)
    t.add_column("value",  justify="right")

    t.add_row("Total runtime",         f"{s['total_runtime_ms']:.3f} ms")
    t.add_row("Total FLOPs",           f"{s['total_flops'] / 1e9:.2f} GFLOPs")
    t.add_row("Total bytes accessed",  f"{s['total_bytes_accessed'] / 1e6:.1f} MB")
    t.add_row("Overall arith. intensity", f"{s['overall_ai']:.2f} FLOPs/byte")
    t.add_row("Avg compute utilization",  f"{s['avg_compute_util']:.1%}")
    t.add_row("Avg cache hit rate",       f"{s['avg_cache_hit_rate']:.1%}")
    t.add_row("Achieved TFLOPS",          f"{s['achieved_tflops']:.4f}")
    t.add_row("Peak (effective) TFLOPS",  f"{s['peak_tflops_hw']:.3f}")
    t.add_row("Memory-bound ops",         str(s["memory_bound_ops"]))
    t.add_row("Compute-bound ops",        str(s["compute_bound_ops"]))

    console.print(t)
