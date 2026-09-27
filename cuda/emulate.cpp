#include <array>
// emulate.cpp — CPU emulation of the exact index arithmetic in matmul.cu.
//
// Cannot test occupancy or memory coalescing, but it does test the thing that
// actually breaks hand-written tiled GEMMs: tile indexing, shared-memory
// addressing, and boundary handling on sizes that are not multiples of the tile.
//
// __syncthreads() is emulated by running every thread through one phase before
// any thread enters the next, which is exactly the barrier's semantics.
//
// Build: g++ -O2 -std=c++14 emulate.cpp -o emulate && ./emulate

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

static void reference(const vector<float>& A, const vector<float>& B,
                      vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[i*K+k] * (double)B[k*N+j];
            C[i*N+j] = (float)acc;
        }
}

// ---- mirrors matmul_tiled<TILE> ----
template <int TILE>
static void emu_tiled(const vector<float>& A, const vector<float>& B,
                      vector<float>& C, int M, int N, int K) {
    const int gridY = (M + TILE - 1) / TILE;
    const int gridX = (N + TILE - 1) / TILE;
    const int numTiles = (K + TILE - 1) / TILE;

    for (int by = 0; by < gridY; ++by)
    for (int bx = 0; bx < gridX; ++bx) {
        vector<float> As(TILE*TILE), Bs(TILE*TILE), acc(TILE*TILE, 0.0f);

        for (int t = 0; t < numTiles; ++t) {
            // phase 1: cooperative load (all threads)
            for (int ty = 0; ty < TILE; ++ty)
            for (int tx = 0; tx < TILE; ++tx) {
                const int row  = by*TILE + ty;
                const int col  = bx*TILE + tx;
                const int aCol = t*TILE + tx;
                const int bRow = t*TILE + ty;
                As[ty*TILE+tx] = (row < M && aCol < K) ? A[row*K + aCol] : 0.0f;
                Bs[ty*TILE+tx] = (bRow < K && col < N) ? B[bRow*N + col] : 0.0f;
            }
            // __syncthreads()
            // phase 2: compute (all threads)
            for (int ty = 0; ty < TILE; ++ty)
            for (int tx = 0; tx < TILE; ++tx)
                for (int k = 0; k < TILE; ++k)
                    acc[ty*TILE+tx] += As[ty*TILE+k] * Bs[k*TILE+tx];
            // __syncthreads()
        }
        for (int ty = 0; ty < TILE; ++ty)
        for (int tx = 0; tx < TILE; ++tx) {
            const int row = by*TILE + ty, col = bx*TILE + tx;
            if (row < M && col < N) C[row*N+col] = acc[ty*TILE+tx];
        }
    }
}

// ---- mirrors matmul_regtiled<BM,BN,BK,TM,TN> ----
template <int BM, int BN, int BK, int TM, int TN>
static void emu_regtiled(const vector<float>& A, const vector<float>& B,
                         vector<float>& C, int M, int N, int K) {
    const int THREADS = (BM*BN)/(TM*TN);
    const int strideA = THREADS / BK;
    const int strideB = THREADS / BN;
    const int gridY = (M + BM - 1) / BM;
    const int gridX = (N + BN - 1) / BN;

    // static invariants the launch config depends on
    static_assert((BM*BN) % (TM*TN) == 0, "block must divide evenly into thread tiles");
    static_assert(((BM*BN)/(TM*TN)) % BK == 0, "THREADS must be a multiple of BK");
    static_assert(((BM*BN)/(TM*TN)) % BN == 0, "THREADS must be a multiple of BN");

    for (int cRow = 0; cRow < gridY; ++cRow)
    for (int cCol = 0; cCol < gridX; ++cCol) {
        vector<float> As(BM*BK), Bs(BK*BN);
        vector<float> results(THREADS * TM * TN, 0.0f);

        for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
            // phase 1: cooperative load
            for (int tid = 0; tid < THREADS; ++tid) {
                const int innerColA = tid % BK, innerRowA = tid / BK;
                const int innerColB = tid % BN, innerRowB = tid / BN;
                for (int off = 0; off < BM; off += strideA) {
                    const int gRow = cRow*BM + innerRowA + off;
                    const int gCol = bkIdx + innerColA;
                    As[(innerRowA+off)*BK + innerColA] =
                        (gRow < M && gCol < K) ? A[gRow*K + gCol] : 0.0f;
                }
                for (int off = 0; off < BK; off += strideB) {
                    const int gRow = bkIdx + innerRowB + off;
                    const int gCol = cCol*BN + innerColB;
                    Bs[(innerRowB+off)*BN + innerColB] =
                        (gRow < K && gCol < N) ? B[gRow*N + gCol] : 0.0f;
                }
            }
            // __syncthreads()
            // phase 2: compute
            for (int tid = 0; tid < THREADS; ++tid) {
                const int threadCol = tid % (BN/TN), threadRow = tid / (BN/TN);
                float regM[TM], regN[TN];
                for (int dot = 0; dot < BK; ++dot) {
                    for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow*TM + i)*BK + dot];
                    for (int j = 0; j < TN; ++j) regN[j] = Bs[dot*BN + threadCol*TN + j];
                    for (int i = 0; i < TM; ++i)
                        for (int j = 0; j < TN; ++j)
                            results[tid*TM*TN + i*TN + j] += regM[i]*regN[j];
                }
            }
            // __syncthreads()
        }
        for (int tid = 0; tid < THREADS; ++tid) {
            const int threadCol = tid % (BN/TN), threadRow = tid / (BN/TN);
            for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j) {
                const int gRow = cRow*BM + threadRow*TM + i;
                const int gCol = cCol*BN + threadCol*TN + j;
                if (gRow < M && gCol < N) C[gRow*N+gCol] = results[tid*TM*TN + i*TN + j];
            }
        }
    }
}

static bool check(const char* name, int M, int N, int K,
                  const vector<float>& got, const vector<float>& want) {
    double worst = 0.0; int bi = -1;
    for (size_t i = 0; i < want.size(); ++i) {
        double d = fabs((double)got[i] - (double)want[i]);
        if (d > worst) { worst = d; bi = (int)i; }
    }
    const double tol = 1e-3;
    const bool ok = worst <= tol;
    printf("  %-9s M=%4d N=%4d K=%4d   max|diff| = %.3e  %s",
           name, M, N, K, worst, ok ? "PASS" : "FAIL");
    if (!ok) printf("   (worst at row %d col %d)", bi/N, bi%N);
    printf("\n");
    return ok;
}

int main() {
    // Deliberately include sizes that are NOT multiples of the tile dims,
    // since that is where boundary logic breaks.
    vector<array<int,3>> cases = {
        {{ 64,  64,  64}},   // exact multiple of both tilings
        {{128, 128, 128}},
        {{100,  70,  53}},   // nothing divides anything
        {{ 33,  65,  17}},   // smaller than one BM/BN tile
        {{  1,   1,   1}},   // degenerate
        {{200, 130,  96}},
        {{ 65,  64,  64}},   // one past a tile boundary in M
        {{ 64,  65,  64}},   // one past in N
        {{ 64,  64,  65}},   // one past in K
    };

    int fails = 0;
    srand(1234);
    for (auto& c : cases) {
        const int M = c[0], N = c[1], K = c[2];
        vector<float> A((size_t)M*K), B((size_t)K*N);
        for (auto& v : A) v = (float)rand()/RAND_MAX - 0.5f;
        for (auto& v : B) v = (float)rand()/RAND_MAX - 0.5f;

        vector<float> ref((size_t)M*N, 0.0f);
        reference(A, B, ref, M, N, K);

        vector<float> c1((size_t)M*N, 0.0f), c2((size_t)M*N, 0.0f);
        emu_tiled<32>(A, B, c1, M, N, K);
        emu_regtiled<64,64,8,4,4>(A, B, c2, M, N, K);

        if (!check("tiled",    M, N, K, c1, ref)) ++fails;
        if (!check("regtiled", M, N, K, c2, ref)) ++fails;
    }
    printf("\n%s  (%d failures)\n", fails ? "SOME CASES FAILED" : "ALL INDEX MATH VERIFIED", fails);
    return fails ? 1 : 0;
}
