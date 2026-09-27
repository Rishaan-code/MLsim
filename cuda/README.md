# SGEMM kernels vs cuBLAS

Three hand-written fp32 matmul kernels for Turing, each fixing the bottleneck the
previous one hit, measured against cuBLAS on the same T4 used for the model
calibration in the parent project.

Build and run:

```bash
nvcc -O3 -arch=sm_75 matmul.cu -o matmul -lcublas && ./matmul
```

Or open `run_on_colab.ipynb` on a Colab T4 and Run All. The source is embedded in
the notebook, so nothing needs uploading.

## Measured on a T4 (GFLOP/s)

| size | naive | tiled | regtiled | cuBLAS |
|---|---|---|---|---|
| 256 | 340 | 504 | 786 | 1062 |
| 512 | 375 | 616 | 1757 | 3303 |
| 1024 | 498 | 928 | **3143** | 5209 |
| 2048 | 438 | 899 | 2501 | 4511 |
| 4096 | 432 | 893 | 2421 | 4176 |

At 1024 the register-tiled kernel reaches 3143 GFLOP/s, 6.3x the naive version
and 60% of cuBLAS, or 39% of the T4's 8.1 TFLOPS fp32 peak. Across the
compute-bound sizes it holds 53-60% of cuBLAS.

Every kernel is checked against cuBLAS for correctness before it is timed. Max
absolute deviation stays at or below 4.2e-05, which is fp32 accumulation-order
noise, not a logic error. The cuBLAS row compares cuBLAS against itself, so its
0.000e+00 is trivially true and is not evidence of anything.

## What each kernel changes

**naive** — one thread per output element, operands read straight from global
memory. Each element of A is re-read N times. Bandwidth bound, and it shows:
throughput is essentially flat at 430-500 GFLOP/s regardless of size.

**tiled** — the block stages a 32x32 tile of A and B in shared memory and every
thread in the block reuses it, cutting global loads by a factor of 32. Roughly
2x naive, plateauing near 900 GFLOP/s. Shared-memory bandwidth is the new wall:
each thread still reads two shared values per single FMA.

**regtiled** — each thread owns a 4x4 patch of C in registers. Per accumulation
step it reads TM + TN = 8 values from shared memory and issues TM * TN = 16
FMAs. That is 2 FMAs per shared read, against 0.5 in the tiled kernel (2 reads
per 1 FMA), so the shared-read-to-FLOP ratio improves 4x.
Block computes a 64x64 output tile with 256 threads, marching over K in steps
of 8. This is where the large win comes from: 6.3x naive.

## Why it is not at cuBLAS parity

cuBLAS uses vectorized 128-bit loads, double-buffered shared memory so the next
tile prefetches during compute on the current one, warp-level tiling matched to
the scheduler, and per-architecture autotuning. None of that is here. 60% of
cuBLAS from a readable kernel is the expected place to land, and closing the
remaining gap is mostly those four techniques.

## Throughput drops past 1024 for every kernel, including cuBLAS

cuBLAS itself falls from 5209 to 4176 GFLOP/s (-20%) between 1024 and 4096, and
regtiled falls -23%. Because the vendor library degrades by a similar fraction,
this is a platform effect rather than a kernel problem: sustained large GEMMs
heat the part and the clock drops. Colab T4s are passively cooled and shared.
Worth knowing before reading anything into the 2048 and 4096 columns.

## Cross-check against the parent project

The parent project measured fp32 matmul through `torch.mm`. This measures cuBLAS
directly. At 4096, where launch and dispatch overhead is negligible next to the
work, the two agree within 2.4% (32.15 ms vs 32.91 ms), which is reassurance
that both harnesses are timing the same thing. At smaller sizes they diverge by
25-30%, since the PyTorch path carries dispatch overhead that a direct cuBLAS
call does not.

## emulate.cpp

The tile index arithmetic is verified on CPU, without a GPU, by
`emulate.cpp`. It mirrors the exact indexing of both tiled kernels and emulates
`__syncthreads()` by running every thread through one phase before any thread
enters the next, then diffs against a reference matmul.

```bash
g++ -O2 -std=c++14 emulate.cpp -o emulate && ./emulate
```

9 size configurations pass, both kernels checked in each, so 18 assertions in
total. They include the ones built to break boundary logic: 100x70x53 where
no dimension divides any tile, 33x65x17 smaller than a single block tile, 1x1x1,
and 65x64x64 / 64x65x64 / 64x64x65 sitting one element past a tile boundary in
each dimension.
