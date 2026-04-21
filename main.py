"""
mlsim — ML accelerator workload simulator
Usage:
    python -m mlsim.main [suite]

Suites: llm, vision, scaling, all  (default: all)
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel   import Panel
from rich         import box

from .core import (
    cpu_desktop_hierarchy, npu_hierarchy, mobile_hierarchy,
    cpu_avx512, npu_systolic, mobile_arm_neon,
    llm_inference_suite, vision_model_suite, matmul_scaling_suite,
)
from .sim      import HardwareConfig, compare_hardware
from .analyze  import compute_speedup_table, memory_pressure_report, print_suite_summary
from .visualize import save_all

console = Console()

OUT_DIR = Path("mlsim_output")


def build_hardware() -> list[HardwareConfig]:
    return [
        HardwareConfig(
            name="CPU (x86 AVX-512)",
            memory=cpu_desktop_hierarchy(freq_ghz=3.5),
            compute=cpu_avx512(freq_ghz=3.5),
        ),
        HardwareConfig(
            name="NPU (systolic + HBM)",
            memory=npu_hierarchy(freq_ghz=1.25),
            compute=npu_systolic(freq_ghz=1.25),
        ),
        HardwareConfig(
            name="Mobile SoC (ARM NEON)",
            memory=mobile_hierarchy(freq_ghz=2.4),
            compute=mobile_arm_neon(freq_ghz=2.4),
        ),
    ]


def run_suite(suite_name: str, workloads, hw_configs):
    console.rule(f"[bold cyan]{suite_name}[/bold cyan]")

    runs = compare_hardware(workloads, hw_configs)

    for name, run in runs.items():
        print_suite_summary(run)

    # Speedup tables vs CPU baseline.
    cpu_run = runs.get("CPU (x86 AVX-512)")
    if cpu_run:
        for name, run in runs.items():
            if name == cpu_run.hardware.name:
                continue
            table = compute_speedup_table(cpu_run, run)
            table.print()

    # Memory pressure.
    console.rule("[bold]Memory pressure[/bold]")
    for name, run in runs.items():
        for report in memory_pressure_report(run):
            color = {"low": "green", "medium": "yellow", "high": "red"}[report.pressure]
            console.print(f"  [{color}]{report}[/{color}]")

    # Save plots.
    out = OUT_DIR / suite_name.lower().replace(" ", "_")
    save_all(list(runs.values()), out_dir=out)
    console.print(f"\n  [dim]Charts saved → {out}[/dim]\n")

    return runs


def main():
    suite_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    hw = build_hardware()

    console.print(Panel(
        "[bold]mlsim — ML accelerator workload simulator[/bold]\n"
        "Cycle-approximate modeling of ML ops across CPU, NPU, and mobile SoC",
        box=box.ROUNDED, style="cyan",
    ))

    suites = {
        "llm":     ("LLM inference (single transformer layer)", llm_inference_suite()),
        "vision":  ("Vision model  (ResNet-50 conv stack)",     vision_model_suite()),
        "scaling": ("MatMul scaling (32³ → 8192³)",             matmul_scaling_suite()),
    }

    if suite_arg == "all":
        for key, (name, workloads) in suites.items():
            run_suite(name, workloads, hw)
    elif suite_arg in suites:
        name, workloads = suites[suite_arg]
        run_suite(name, workloads, hw)
    else:
        console.print(f"[red]Unknown suite '{suite_arg}'. Choose: llm, vision, scaling, all[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
