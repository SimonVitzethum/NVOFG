// Tensor-Core feasibility spike (design analysis §C/§D): prove that a CUDA WMMA
// kernel actually executes on the Tensor Cores of this GPU and returns a correct
// GEMM. This is the compute half of the future CUDA CNN interpolator backend; the
// Vulkan<->CUDA interop half is proven separately (cuda_vk_interop.cu).
//
// Builds only when NVOFG_ENABLE_CUDA is set. Verifies C = A*B (fp16 inputs, fp32
// accumulate) against a CPU reference; exit 0 = Tensor Cores produced correct output.

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nvcuda;

static constexpr int M = 64, N = 64, K = 64;   // multiples of 16 (WMMA tile)
static constexpr int T = 16;                    // WMMA 16x16x16

// One warp per 16x16 output tile; grid = (M/16, N/16).
__global__ void wmma_gemm(const half* A, const half* B, float* C) {
    int warpM = blockIdx.x;   // tile row
    int warpN = blockIdx.y;   // tile col

    wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, T, T, T, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += T) {
        wmma::load_matrix_sync(a_frag, A + (warpM * T) * K + k, K);   // A tile, ld=K
        wmma::load_matrix_sync(b_frag, B + k * N + (warpN * T), N);   // B tile, ld=N
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);              // Tensor-Core MMA
    }
    wmma::store_matrix_sync(C + (warpM * T) * N + (warpN * T), c_frag, N, wmma::mem_row_major);
}

#define CUDA_OK(e)                                                              \
    do {                                                                        \
        cudaError_t _r = (e);                                                   \
        if (_r != cudaSuccess) {                                                \
            std::fprintf(stderr, "%s -> %s\n", #e, cudaGetErrorString(_r));     \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDevice(&dev));
    CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    std::printf("CUDA device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    std::vector<half> hA(M * K), hB(K * N);
    std::vector<float> hC(M * N, 0.0f), ref(M * N, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) hA[i * K + k] = __float2half(float((i + k) % 3));
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j) hB[k * N + j] = __float2half(float((k + j) % 3));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += float((i + k) % 3) * float((k + j) % 3);
            ref[i * N + j] = acc;
        }

    half *dA, *dB;
    float* dC;
    CUDA_OK(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CUDA_OK(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CUDA_OK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));

    dim3 grid(M / T, N / T);
    wmma_gemm<<<grid, 32>>>(dA, dB, dC);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int i = 0; i < M * N; ++i) maxErr = std::max(maxErr, (double)std::abs(hC[i] - ref[i]));
    std::printf("WMMA GEMM %dx%dx%d: max abs error vs CPU = %.3f\n", M, N, K, maxErr);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    bool ok = maxErr < 1e-3;
    std::printf("RESULT: %s (Tensor Cores %s)\n", ok ? "PASS" : "FAIL",
                ok ? "executed correctly" : "mismatch");
    return ok ? 0 : 1;
}
