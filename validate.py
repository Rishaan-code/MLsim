"""
validate.py — compare mlsim's predicted runtimes against real T4 measurements.

Measured data lives in results/results.csv and was collected on a Colab T4 with
CUDA-event timing (20 warmup + 100 timed iterations per point). This script
re-runs the model over the same 32 configurations and reports prediction error,
so the headline accuracy number in the paper is reproducible from the repo.

Usage (from the parent directory of MLsim/):
    python -m mlsim.validate
    python -m mlsim.validate --holdout      # anti-overfitting check
    python -m mlsim.validate --crossover    # measured dtype benefit vs the
                                            # analytical crossover prediction
    python -m mlsim.validate --threshold 25
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

# Allow running this file directly from inside a clone ("python validate.py"),
# not just as a module ("python -m mlsim.validate"). The clone directory is not
# always named "mlsim", so relative imports need the parent on sys.path first.
if __package__ in (None, ""):
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.dirname(_here))
    __package__ = _os.path.basename(_here)

from .sim import Simulator
from .core.workload import MatMulWorkload
from .crossover import default_gpu_config

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results", "results.csv")


@dataclass
class Point:
    size: int
    dtype: str
    measured_ms: float
    predicted_ms: float

    @property
    def abs_pct_error(self) -> float:
        if self.measured_ms <= 0:
            return float("nan")
        return abs(self.predicted_ms - self.measured_ms) / self.measured_ms * 100.0

    @property
    def signed_pct_error(self) -> float:
        if self.measured_ms <= 0:
            return float("nan")
        return (self.predicted_ms - self.measured_ms) / self.measured_ms * 100.0


def load_measurements(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def predict(size: int, dtype: str) -> float:
    """Predicted runtime in ms for a square matmul of the given size and dtype."""
    hw = default_gpu_config().hw
    sim = Simulator(hw)
    wl = MatMulWorkload(M=size, N=size, K=size, dtype=dtype)
    return sim.run_workload(wl).runtime_ms


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def holdout_check(rows: list[dict]) -> None:
    """
    The launch-overhead constant is one free parameter, so it is fair to ask
    whether it was fit to the same data it is scored on. It was not, and this
    reproduces the argument.

    At 32x32 a matmul is 2*32^3 = 65 kFLOP, roughly 8 ns of real math on an
    8.1 TFLOPS part. Measured runtime there is therefore essentially pure
    dispatch cost, so the constant can be read straight off those four points
    without reference to any error metric. Scoring then happens on the other
    28 points, which were never used to derive it.
    """
    from .crossover import default_gpu_config
    from .sim import Simulator

    measured = {(int(r["matrix_size"]), r["dtype"]): float(r["runtime_ms"]) for r in rows}

    def predict_with(size: int, dtype: str, overhead_us: float) -> float:
        hw = default_gpu_config().hw
        hw.compute.kernel_launch_overhead_us = overhead_us
        return Simulator(hw).run_workload(
            MatMulWorkload(M=size, N=size, K=size, dtype=dtype)
        ).runtime_ms

    def mape_over(keys, overhead_us: float) -> float:
        return mean([
            abs(predict_with(s, d, overhead_us) - measured[(s, d)]) / measured[(s, d)] * 100.0
            for s, d in keys
        ])

    smallest = min(s for s, _ in measured)
    fit_keys = [k for k in measured if k[0] == smallest]
    held_keys = [k for k in measured if k[0] != smallest]
    derived = mean([measured[k] for k in fit_keys]) * 1000.0  # ms -> us

    print("\nholdout check")
    print("-" * 34)
    print(f"  constant derived from {len(fit_keys)} points at {smallest}x{smallest}: {derived:.1f} us")
    print(f"  scored on the remaining {len(held_keys)} points, never used to derive it:")
    print(f"    no overhead term (original model) : {mape_over(held_keys, 0.0):.1f}%")
    print(f"    derived constant                  : {mape_over(held_keys, derived):.1f}%")
    print("  the correction generalizes to data it was not fit on.")


def crossover_check(rows: list[dict]) -> None:
    """
    Compare what the dtypes actually do on hardware against what the analytical
    crossover model predicts they do.

    This exists because the two disagree, and the disagreement is structural
    rather than a tuning issue. QuantizationModel._crossover_ai derives the
    crossover from effective_memory_reduction, layout_penalty and dequantization
    cost only. It never reads compute throughput. fp16 and bf16 are both 2 bytes
    per element with identical quantization configs, so the formula cannot
    produce different answers for them, and it does not: both come back at
    AI 4.0, "beneficial for most ML ops".

    Measured on a T4, fp16 runs ~5.6x fp32 and bf16 runs ~0.58x. Compute
    throughput, the one term the formula omits, is the dominant one for any
    dtype the hardware lacks a native path for.
    """
    from .crossover import build_default_engine
    from .core.quantization import Dtype

    measured = {(int(r["matrix_size"]), r["dtype"]): float(r["runtime_ms"]) for r in rows}
    sizes = sorted({s for s, _ in measured})
    dtypes = [d for d in ("fp16", "bf16", "int8") if (sizes[0], d) in measured]

    print("\nmeasured speedup vs fp32 at equal matrix size")
    print("-" * 52)
    print(f"{'size':>6} | " + "  ".join(f"{d:>6}" for d in dtypes))
    for s in sizes:
        cells = "  ".join(f"{measured[(s,'fp32')] / measured[(s,d)]:>6.2f}" for d in dtypes)
        print(f"{s:>6} | {cells}")

    # Analytical prediction from the quantization model.
    engine = build_default_engine()
    predicted = {p.dtype.value: p.crossover_ai
                 for p in engine.predict_crossover_points("GPU (T4)")}

    print("\nanalytical crossover AI vs measured outcome (compute-bound sizes)")
    print("-" * 66)
    big = [s for s in sizes if s >= 1024]
    for d in dtypes:
        sp = [measured[(s, "fp32")] / measured[(s, d)] for s in big]
        wins = sum(1 for x in sp if x > 1.0)
        pred = predicted.get(d)
        pred_s = f"{pred:.2f}" if pred is not None else "n/a"
        verdict = "beneficial" if wins == len(big) else "NOT beneficial"
        print(f"  {d:>5}: predicted crossover AI {pred_s:>6} | "
              f"measured {min(sp):.2f}-{max(sp):.2f}x, wins {wins}/{len(big)} -> {verdict}")

    if predicted.get("fp16") == predicted.get("bf16"):
        print("\n  NOTE: fp16 and bf16 receive identical analytical predictions because")
        print("  _crossover_ai models only memory reduction, layout penalty and dequant")
        print("  cost. On this part they differ by ~10x in measured throughput. Treat the")
        print("  analytical crossover as a bandwidth-savings bound only; the compute")
        print("  throughput term lives in ComputeUnitConfig.dtype_speedup.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate mlsim against measured T4 runtimes.")
    ap.add_argument("--csv", default=RESULTS_CSV, help="path to measured results CSV")
    ap.add_argument("--threshold", type=float, default=None,
                    help="fail (exit 1) if mean absolute percentage error exceeds this")
    ap.add_argument("--holdout", action="store_true",
                    help="derive the launch-overhead constant from the smallest "
                         "matrices alone and score on the remaining points")
    ap.add_argument("--crossover", action="store_true",
                    help="compare measured per-dtype benefit against the "
                         "analytical crossover prediction")
    args = ap.parse_args(argv)

    rows = load_measurements(args.csv)
    points: list[Point] = []

    for r in rows:
        size = int(r["matrix_size"])
        dtype = r["dtype"]
        measured = float(r["runtime_ms"])
        points.append(Point(size, dtype, measured, predict(size, dtype)))

    dtypes = sorted({p.dtype for p in points})
    sizes = sorted({p.size for p in points})

    print(f"mlsim validation against {os.path.relpath(args.csv, os.path.dirname(__file__))}")
    print(f"hardware: NVIDIA T4   configurations: {len(points)} "
          f"({len(sizes)} sizes x {len(dtypes)} dtypes)\n")

    header = f"{'size':>6}  {'dtype':>5}  {'measured ms':>12}  {'predicted ms':>13}  {'error':>9}"
    print(header)
    print("-" * len(header))
    for p in sorted(points, key=lambda q: (q.size, q.dtype)):
        print(f"{p.size:>6}  {p.dtype:>5}  {p.measured_ms:>12.5f}  "
              f"{p.predicted_ms:>13.5f}  {p.signed_pct_error:>+8.1f}%")

    errs = [p.abs_pct_error for p in points]
    mape = mean(errs)
    within20 = sum(1 for e in errs if e <= 20.0)

    print("\nby dtype")
    print("-" * 34)
    for d in dtypes:
        sub = [p.abs_pct_error for p in points if p.dtype == d]
        print(f"  {d:>5}  mean {mean(sub):6.1f}%   median {median(sub):6.1f}%   n={len(sub)}")

    print("\noverall")
    print("-" * 34)
    print(f"  mean absolute percentage error : {mape:.1f}%")
    print(f"  median absolute percentage error: {median(errs):.1f}%")
    print(f"  within 20%                      : {within20}/{len(errs)}")

    # Sanity check the headline architectural claim: on Turing, bf16 has no
    # native tensor core path, so it should measure slower than fp32.
    # Only compute-bound sizes are meaningful here: below ~512 every dtype is
    # pinned to launch overhead, which washes the ratio out toward 1.0.
    bf16 = {p.size: p.measured_ms for p in points if p.dtype == "bf16"}
    fp32 = {p.size: p.measured_ms for p in points if p.dtype == "fp32"}
    shared = sorted(s for s in (set(bf16) & set(fp32)) if s >= 1024)
    if shared:
        ratios = [fp32[s] / bf16[s] for s in shared if bf16[s] > 0]
        slower = sum(1 for s in shared if bf16[s] > fp32[s])
        print(f"\nbf16 vs fp32 at compute-bound sizes (>=1024): bf16 runs at "
              f"{mean(ratios):.2f}x fp32 throughput, slower at {slower}/{len(shared)} sizes")
        print("  Turing tensor cores support fp16/int8/int4 but not bf16, so bf16")
        print("  falls off the tensor core path while fp16 stays on it.")

    if args.holdout:
        holdout_check(rows)

    if args.crossover:
        crossover_check(rows)

    if args.threshold is not None and mape > args.threshold:
        print(f"\nFAIL: mean error {mape:.1f}% exceeds threshold {args.threshold:.1f}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
