# mlsim
**Paper:** https://doi.org/10.5281/zenodo.19685708

### Relationship to the paper

The paper is mlsim v0.2 (April 2026) and reports **18.7% mean error at large
matrix sizes**. It also lists, as a known limitation, that the model does not
account for CUDA kernel launch overhead: "mlsim is cycle-approximate, not
cycle-accurate. It does not model CUDA kernel launch overhead."

This repository implements that missing term. A measured per-launch dispatch
cost brings error to **17.4% mean / 11.3% median across all 32 measured points**
rather than large sizes only, and the constant is derived on the four smallest
matrices and scored on the other 28 (`python validate.py --holdout`), so it is
a correction rather than a fit. The two numbers are not in conflict: they are
different model versions measured over different size ranges.

The paper's three empirical findings are unchanged, since they come from the
measurements rather than the model.


A first-order performance model for ML accelerator workloads. The core idea is simple: before you start benchmarking fp16 vs int8 on real hardware, you should be able to predict whether switching dtypes will actually help based on the hardware specs alone.

Most people pick dtypes by running experiments on every target. That works, but it tells you nothing that generalizes to the next target. mlsim models the roofline, the memory hierarchy, and the actual overhead of quantization (dequantization cost, layout penalties, scale storage) to predict the arithmetic intensity threshold where a dtype switch goes from helpful to harmful.

I validated it against real T4 measurements (on a free Colab GPU) and found things that naive roofline analysis misses entirely: bf16 is slower than fp32 on T4 because there's no native tensor core support for it, and the int8 crossover point is 62x higher than theory predicts. The paper in `/paper` goes into the full findings.

## What it models

- Memory hierarchies: L1/L2/L3/HBM/DRAM with realistic latencies and bandwidths
- Compute units: x86 AVX-512, systolic array (NPU/TPU), ARM NEON, GPU tensor cores
- Workloads: MatMul, Conv2D, Attention, Elementwise ops
- Quantization overhead: dequantization cost, layout penalties, scale storage per dtype per hardware
- Roofline analysis: arithmetic intensity, compute vs memory bound classification, crossover prediction

## Setup

```bash
pip install -r requirements.txt
```

## Running the simulator

From inside the clone:

```bash
python main.py llm        # transformer decoder layer workloads
python main.py vision     # ResNet conv stack
python main.py scaling    # matmul from 32x32 to 8192x8192
python main.py all        # everything
```

## Using it programmatically

```python
from mlsim import MatMulWorkload, Simulator, HardwareConfig
from mlsim.core import cpu_desktop_hierarchy, cpu_avx512

hw = HardwareConfig("my CPU", cpu_desktop_hierarchy(), cpu_avx512())
wl = MatMulWorkload(M=1024, N=1024, K=512, dtype="fp16")
result = Simulator(hw).run_workload(wl)

print(result.runtime_ms)   # predicted runtime
print(result.bottleneck)   # "compute" or "memory"
```

## Validating against real hardware

`results/results.csv` holds 32 measured matmul runtimes from a Colab T4 (8 sizes x 4 dtypes),
collected with CUDA-event timing and 20 warmup + 100 timed iterations per point.
To re-run the model over those same configurations and print the prediction error:

```bash
python validate.py              # per-point table + mean/median error
python validate.py --holdout    # derive the launch-overhead constant from the
                                # smallest matrices only, score on the rest
python validate.py --threshold 25   # non-zero exit if error regresses past 25%
```

Two things worth knowing about the model's validity region:

- Above 1024x1024 it predicts within roughly 5-8%.
- Below about 512 it is dominated by fixed kernel launch cost rather than FLOPs. A
  32x32 matmul is 65 kFLOP, about 8 ns of real math on a T4, but measures 16-27 us.
  Without a dispatch-cost term the model underpredicts those points by ~99%. The
  `kernel_launch_overhead_us` field handles this; see the holdout check for why that
  constant is a measured quantity rather than a tuned one.

## Hand-written CUDA kernels

`cuda/` holds three fp32 SGEMM kernels for Turing (naive, shared-memory tiled,
register tiled) benchmarked against cuBLAS on the same T4 used to calibrate this
model. The register-tiled kernel reaches 3143 GFLOP/s at 1024x1024, 6.3x the naive
version and 60% of cuBLAS. See `cuda/README.md`.

## Known limitation of the analytical crossover

The crossover AI returned by `QuantizationModel._crossover_ai` is derived from memory
reduction, layout penalty and dequantization cost only. It does not read compute
throughput, so it cannot tell two dtypes of the same width apart. fp16 and bf16 are
both 2 bytes here and both come back at AI 4.0.

The measurements in `results/results.csv` say otherwise:

| dtype | analytical crossover AI | measured speedup vs fp32 (sizes >= 1024) | wins |
|---|---|---|---|
| fp16 | 4.00 | 5.54-5.72x | 3/3 |
| bf16 | 4.00 | 0.55-0.62x | 0/3 |
| int8 | 2.74 | 4.05-5.05x | 3/3 |

Turing has a native fp16 tensor core path and no bf16 one, so bf16 falls back to a
slower route. For any dtype the hardware lacks native support for, compute throughput
is the dominant term, and it is exactly the term the analytical formula omits. The
throughput term lives in `ComputeUnitConfig.dtype_speedup` and is applied in
`ComputeSimulator.simulate`.

Reproduce with:

```bash
python validate.py --crossover
```

## Running the crossover analysis

```python
from mlsim.crossover import build_default_engine

engine = build_default_engine()
engine.print_crossover_summary("GPU (T4)")
engine.print_crossover_summary("CPU (x86 AVX-512)")
```

## Plugging in your own hardware

```python
from mlsim.core import MemoryHierarchy, CacheConfig, MemLevel
from mlsim.core.compute import ComputeUnitConfig, ComputeArch
from mlsim.sim import HardwareConfig, Simulator
from mlsim.core.workload import MatMulWorkload

mem = MemoryHierarchy(
    name="my chip",
    freq_ghz=1.5,
    levels={
        MemLevel.L1:   CacheConfig(size_kb=128,  line_bytes=64, associativity=8,
                                   hit_latency_cycles=4, bandwidth_gb_s=3000,
                                   miss_penalty_cycles=12),
        MemLevel.DRAM: CacheConfig(size_kb=16*1024*1024, line_bytes=64, associativity=1,
                                   hit_latency_cycles=200, bandwidth_gb_s=100,
                                   miss_penalty_cycles=0),
    }
)

compute = ComputeUnitConfig(
    arch=ComputeArch.SYSTOLIC,
    freq_ghz=1.5,
    num_cores=512,
    simd_width_floats=16,
    peak_tflops=12.0,
    mac_efficiency=0.60,
    dtype_speedup={"fp32": 1.0, "fp16": 2.0, "int8": 4.0},
)

hw  = HardwareConfig("my chip", mem, compute)
wl  = MatMulWorkload(M=2048, N=2048, K=2048, dtype="fp16")
res = Simulator(hw).run_workload(wl)
print(res.runtime_ms, res.bottleneck)
```

## Validating against real hardware

The `experiments/benchmark.ipynb` notebook runs the same matmul sweep on a real GPU (tested on Colab T4). Drop the resulting `results.csv` into `results/` and the crossover engine will calibrate against it automatically.

## Repo structure

```
mlsim/
├── core/               memory, compute, roofline, workload, quantization models
├── experiments/        Colab benchmark notebook
├── results/            real T4 hardware measurements
├── paper/              writeup with findings
├── sim.py              main simulation engine
├── crossover.py        quantization crossover analysis
├── analyze.py          metrics and speedup tables
├── visualize.py        roofline plots and charts
└── main.py             CLI entrypoint
```

## Paper

The full writeup is in `paper/mlsim_paper.pdf`. Short version: hardware dtype support matters more than memory reduction ratio, bf16 on T4 is a trap, and the int8 crossover point is much higher than roofline theory predicts.
