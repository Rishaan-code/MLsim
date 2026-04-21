# mlsim

A cycle-approximate simulator for ML accelerator workloads. The core idea is simple: before you start benchmarking fp16 vs int8 on real hardware, you should be able to predict whether switching dtypes will actually help based on the hardware specs alone.

Most people pick dtypes by running experiments on every target. That works but it lowkey tells you nothing generalizable. mlsim models the roofline, the memory hierarchy, and the actual overhead of quantization (dequantization cost, layout penalties, scale storage) to predict the arithmetic intensity threshold where a dtype switch goes from helpful to harmful.

I validated it against real T4 GPU measurements (using Google colab free GPU because i no have gpu) and found some things that naive roofline analysis misses entirely: bf16 is slower than fp32 on T4 because there's no native tensor core support for it, and the int8 crossover point is 62x higher than theory predicts. The paper in `/paper` goes into the full findings.

## What it models

- Memory hierarchies: L1/L2/L3/HBM/DRAM with realistic latencies and bandwidths
- Compute units: x86 AVX-512, systolic array (NPU/TPU), ARM NEON, GPU tensor cores
- Workloads: MatMul, Conv2D, Attention, Elementwise ops
- Quantization overhead: dequantization cost, layout penalties, scale storage per dtype per hardware
- Roofline analysis: arithmetic intensity, compute vs memory bound classification, crossover prediction

## Setup

```bash
pip install rich matplotlib numpy plotly
```

## Running the simulator

From the parent directory of `mlsim/`:

```bash
python -m mlsim.main llm        # transformer decoder layer workloads
python -m mlsim.main vision     # ResNet conv stack
python -m mlsim.main scaling    # matmul from 32x32 to 8192x8192
python -m mlsim.main all        # everything
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
