// matmul.cu — hand-written SGEMM kernels benchmarked against cuBLAS.
//
// Three kernels, each fixing the bottleneck the previous one hit:
//   1. naive      — one thread per output, every operand read from global memory
//   2. tiled      — stage tiles in shared memory so each operand is reused BM times
//   3. regtiled   — each thread computes a TM x TN block in registers, which cuts
//                   shared-memory traffic and raises arithmetic intensity per thread
//
// Timing uses CUDA events with warmup, matching the methodology in
// experiments/benchmark.ipynb. Every kernel is checked against cuBLAS before it
// is timed, because a fast wrong kernel is worth nothing.
//
// Build:  nvcc -O3 -arch=sm_75 matmul.cu -o matmul -lcublas
// Run:    ./matmul            (defaults to 256..4096)
//         ./matmul 1024       (single size)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do {                                                     \
    cudaError_t err__ = (x);                                                   \
    if (err__ != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                            \
                cudaGetErrorString(err__), __FILE__, __LINE__);                \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define CUBLAS_CHECK(x) do {                                                   \
    cublasStatus_t st__ = (x);                                                 \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n",                          \
                (int)st__, __FILE__, __LINE__);                                \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// ---------------------------------------------------------------------------
// Kernel 1: naive. Each thread walks the full K dimension out of global memory.
// Every element of A is re-read N times and every element of B re-read M times,
// so this is bandwidth bound and nowhere near peak.
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: shared-memory tiling. Each block cooperatively stages a TILE x TILE
// block of A and of B into shared memory, then every thread in the block reuses
// those staged values. Global loads drop by a factor of TILE.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        // Zero-pad out-of-range loads so the inner loop needs no bounds checks.
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();   // do not overwrite the tile while peers still read it
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 3: register tiling. Each thread now owns a TM x TN patch of C held in
// registers. Per multiply-accumulate the thread reads TM + TN values from shared
// memory and performs TM * TN FMAs, so the shared-memory-to-FLOP ratio improves
// by roughly (TM*TN)/(TM+TN) over kernel 2.
//
// Block computes a BM x BN output tile, marching over K in steps of BK.
// Threads per block = (BM*BN)/(TM*TN).
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_regtiled(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int tid  = threadIdx.x;
    constexpr int THREADS = (BM * BN) / (TM * TN);

    // Where this thread's TM x TN output patch sits inside the BM x BN tile.
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);

    // Cooperative-load indexing. A tile is BM x BK, B tile is BK x BN.
    const int innerColA = tid % BK;
    const int innerRowA = tid / BK;
    constexpr int strideA = THREADS / BK;

    const int innerColB = tid % BN;
    const int innerRowB = tid / BN;
    constexpr int strideB = THREADS / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        #pragma unroll
        for (int off = 0; off < BM; off += strideA) {
            const int gRow = cRow * BM + innerRowA + off;
            const int gCol = bkIdx + innerColA;
            As[(innerRowA + off) * BK + innerColA] =
                (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            const int gRow = bkIdx + innerRowB + off;
            const int gCol = cCol * BN + innerColB;
            Bs[(innerRowB + off) * BN + innerColB] =
                (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gRow = cRow * BM + threadRow * TM + i;
            const int gCol = cCol * BN + threadCol * TN + j;
            if (gRow < M && gCol < N) C[gRow * N + gCol] = threadResults[i * TN + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
static float timeKernel(void (*launch)(const float*, const float*, float*, int, int, int),
                        const float* dA, const float* dB, float* dC,
                        int M, int N, int K, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

static void launch_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
}

static void launch_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<TILE><<<grid, block>>>(A, B, C, M, N, K);
}

static void launch_regtiled(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    constexpr int THREADS = (BM * BN) / (TM * TN);   // 256
    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_regtiled<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

static cublasHandle_t g_handle;
static void launch_cublas(const float* A, const float* B, float* C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major. Computing C^T = B^T * A^T in its terms gives us
    // row-major C without transposing anything in memory.
    cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

static double maxAbsDiff(const std::vector<float>& x, const std::vector<float>& y) {
    double m = 0.0;
    for (size_t i = 0; i < x.size(); ++i) m = fmax(m, fabs((double)x[i] - (double)y[i]));
    return m;
}

int main(int argc, char** argv) {
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    if (argc > 1) { sizes.clear(); sizes.push_back(atoi(argv[1])); }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    CUBLAS_CHECK(cublasCreate(&g_handle));

    printf("%6s  %-10s  %10s  %9s  %9s  %12s\n",
           "size", "kernel", "time(ms)", "GFLOP/s", "vs cuBLAS", "max|diff|");
    printf("--------------------------------------------------------------------\n");

    FILE* csv = fopen("cuda_results.csv", "w");
    fprintf(csv, "size,kernel,time_ms,gflops,frac_of_cublas,max_abs_diff\n");

    for (int n : sizes) {
        const int M = n, N = n, K = n;
        const size_t bytesA = (size_t)M * K * sizeof(float);
        const size_t bytesB = (size_t)K * N * sizeof(float);
        const size_t bytesC = (size_t)M * N * sizeof(float);

        std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
        for (auto& v : hA) v = (float)rand() / RAND_MAX - 0.5f;
        for (auto& v : hB) v = (float)rand() / RAND_MAX - 0.5f;

        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));
        CUDA_CHECK(cudaMalloc(&dRef, bytesC));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

        // Reference result from cuBLAS.
        launch_cublas(dA, dB, dRef, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> hRef((size_t)M * N);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytesC, cudaMemcpyDeviceToHost));

        const double flops = 2.0 * M * N * K;
        const int warmup = 5, iters = (n >= 2048 ? 20 : 50);

        struct Entry { const char* name; void (*fn)(const float*, const float*, float*, int, int, int); };
        Entry entries[] = {
            {"naive",    launch_naive},
            {"tiled",    launch_tiled},
            {"regtiled", launch_regtiled},
            {"cublas",   launch_cublas},
        };

        // Time cuBLAS first so every row can report a fraction against it.
        // (Earlier version computed this inside the loop with cuBLAS last, so
        // the other three kernels always divided by zero and reported 0.00x.)
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        const double cublas_ms = timeKernel(launch_cublas, dA, dB, dC, M, N, K, warmup, iters);

        for (const auto& e : entries) {
            CUDA_CHECK(cudaMemset(dC, 0, bytesC));
            e.fn(dA, dB, dC, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> hC((size_t)M * N);
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
            const double diff = maxAbsDiff(hC, hRef);

            // fp32 accumulation order differs between kernels, so exact equality
            // is not expected. Tolerance scales with K.
            const double tol = 1e-3 * K / 1024.0 + 1e-4;
            const bool ok = diff <= tol;

            const float ms = timeKernel(e.fn, dA, dB, dC, M, N, K, warmup, iters);
            const double gflops = flops / (ms * 1e-3) / 1e9;
            const double frac = cublas_ms / ms;   // 1.0 means parity with cuBLAS

            printf("%6d  %-10s  %10.4f  %9.1f  %8.2fx  %12.2e%s\n",
                   n, e.name, ms, gflops, frac, diff, ok ? "" : "   MISMATCH");
            fprintf(csv, "%d,%s,%.6f,%.2f,%.4f,%.3e\n", n, e.name, ms, gflops, frac, diff);
        }
        printf("\n");

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dRef));
    }

    fclose(csv);
    CUBLAS_CHECK(cublasDestroy(g_handle));
    printf("wrote cuda_results.csv\n");
    return 0;
}
