#include "cnn_cuda.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <map>
#include <string>

namespace nvofg {
namespace {

#define CKr(x) do{ auto e=(x); if(e){ return {}; } }while(0)
#define FR(p) cudaFreeAsync(p,0)

__global__ void k_im2col(const __half* in, __half* col, int Ci, int H, int W, int Ho, int Wo, int st) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; int N = Ho * Wo; if (p >= N) return; int oy = p / Wo, ox = p % Wo;
    for (int ci = 0; ci < Ci; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
        int iy = oy * st - 1 + ky, ix = ox * st - 1 + kx;
        __half v = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? in[(ci * H + iy) * W + ix] : __float2half(0.f);
        col[((ci * 3 + ky) * 3 + kx) * N + p] = v; } }
__global__ void k_gated(const __half* a, const __half* fb, const __half* g, const __half* gb, __half* o, int C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= C * N) return; int c = i / N;
    float av = __half2float(a[i]) + __half2float(fb[c]), gv = __half2float(g[i]) + __half2float(gb[c]);
    o[i] = __float2half((av > 0 ? av : 0.1f * av) * (1.f / (1.f + expf(-gv)))); }
__global__ void k_bias(const __half* a, const __half* b, __half* o, int C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= C * N) return; o[i] = __float2half(__half2float(a[i]) + __half2float(b[i / N])); }
__global__ void k_up(const __half* in, __half* o, int C, int H, int W, int Ho, int Wo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= C * Ho * Wo) return; int c = idx / (Ho * Wo), r = idx % (Ho * Wo), oy = r / Wo, ox = r % Wo;
    float iy = (oy + 0.5f) * H / Ho - 0.5f, ix = (ox + 0.5f) * W / Wo - 0.5f; int y0 = floorf(iy), x0 = floorf(ix); float fy = iy - y0, fx = ix - x0;
    int y0c = max(0, min(H - 1, y0)), y1c = max(0, min(H - 1, y0 + 1)), x0c = max(0, min(W - 1, x0)), x1c = max(0, min(W - 1, x0 + 1));
    const __half* pp = in + (size_t)c * H * W;
    o[idx] = __float2half(__half2float(pp[y0c * W + x0c]) * (1 - fx) * (1 - fy) + __half2float(pp[y0c * W + x1c]) * fx * (1 - fy)
           + __half2float(pp[y1c * W + x0c]) * (1 - fx) * fy + __half2float(pp[y1c * W + x1c]) * fx * fy); }

struct T { __half* d; int C, H, W; };
cublasHandle_t g_cb = nullptr;
struct Cache { std::map<std::string, __half*> wt; std::map<std::string, std::vector<int>> wd; };
std::map<const CnnModel*, Cache> g_cache;

T galloc(int C, int H, int W) { __half* p = nullptr; cudaMallocAsync(&p, sizeof(__half) * C * H * W, 0); return {p, C, H, W}; }
void gemm(const __half* col, const __half* w, __half* out, int Cout, int K, int N) {  // Tensor Cores, fp32 accum
    float al = 1.f, be = 0.f;
    cublasGemmEx(g_cb, CUBLAS_OP_N, CUBLAS_OP_N, N, Cout, K, &al, col, CUDA_R_16F, N, w, CUDA_R_16F, K, &be,
                 out, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
T conv_raw(Cache& c, T in, const std::string& wn, int st) {
    int Cout = c.wd[wn][0], K = in.C * 9, Ho = (in.H + 2 - 3) / st + 1, Wo = (in.W + 2 - 3) / st + 1, N = Ho * Wo;
    __half* col = nullptr; cudaMallocAsync(&col, sizeof(__half) * K * N, 0);
    k_im2col<<<(N + 255) / 256, 256>>>(in.d, col, in.C, in.H, in.W, Ho, Wo, st);
    T o = galloc(Cout, Ho, Wo); gemm(col, c.wt[wn], o.d, Cout, K, N); FR(col); return o;
}
T gated(Cache& c, T in, const std::string& p, int st) {
    T a = conv_raw(c, in, p + ".feat.weight", st), g = conv_raw(c, in, p + ".gate.weight", st);
    int C = a.C, N = a.H * a.W; T o = galloc(C, a.H, a.W);
    k_gated<<<(C * N + 255) / 256, 256>>>(a.d, c.wt[p + ".feat.bias"], g.d, c.wt[p + ".gate.bias"], o.d, C, N);
    FR(a.d); FR(g.d); return o;
}
T seq2(Cache& c, T in, const std::string& e, int st) { T a = gated(c, in, e + ".0", st); T b = gated(c, a, e + ".1", 1); FR(a.d); return b; }
T up(T in, int Ho, int Wo) { T o = galloc(in.C, Ho, Wo); k_up<<<(in.C * Ho * Wo + 255) / 256, 256>>>(in.d, o.d, in.C, in.H, in.W, Ho, Wo); return o; }
T cat(T a, T b) { int N = a.H * a.W; T o = galloc(a.C + b.C, a.H, a.W);
    cudaMemcpyAsync(o.d, a.d, sizeof(__half) * a.C * N, cudaMemcpyDeviceToDevice, 0);
    cudaMemcpyAsync(o.d + a.C * N, b.d, sizeof(__half) * b.C * N, cudaMemcpyDeviceToDevice, 0); return o; }

Cache& ensure(const CnnModel& m) {
    auto it = g_cache.find(&m);
    if (it != g_cache.end()) return it->second;
    Cache& c = g_cache[&m];
    for (auto& kv : m.tensors) {
        __half* p = nullptr; cudaMalloc(&p, sizeof(__half) * kv.second.half.size());
        cudaMemcpy(p, kv.second.half.data(), sizeof(__half) * kv.second.half.size(), cudaMemcpyHostToDevice);
        c.wt[kv.first] = p; c.wd[kv.first] = kv.second.dims;
    }
    return c;
}

void ensureInit() {
    if (g_cb) return;
    cublasCreate(&g_cb); cublasSetMathMode(g_cb, CUBLAS_TENSOR_OP_MATH);
    cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, 0); uint64_t thr = ~0ull;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
}

// Core forward on DEVICE pointers: d_in=[12,H,W] half, d_out=[3,H,W] half. Neither is freed — the
// caller owns them (for the in-pipeline path they are Vulkan-shared external memory). Runs on the
// given stream (0 = the interop stream after it has waited on the Vulkan timeline semaphore).
void fusionCore(Cache& c, const __half* d_in, __half* d_out, int H, int W) {
    T in{const_cast<__half*>(d_in), 12, H, W};
    T s1 = seq2(c, in, "fusion.e1", 1), s2 = seq2(c, s1, "fusion.e2", 2), s3 = seq2(c, s2, "fusion.e3", 2), s4 = seq2(c, s3, "fusion.e4", 2);
    T u3 = up(s4, s3.H, s3.W), c3 = cat(u3, s3), d3 = seq2(c, c3, "fusion.d3", 1);
    T u2 = up(d3, s2.H, s2.W), c2 = cat(u2, s2), d2 = seq2(c, c2, "fusion.d2", 1);
    T u1 = up(d2, s1.H, s1.W), c1 = cat(u1, s1), d1 = seq2(c, c1, "fusion.d1", 1);
    T raw = conv_raw(c, d1, "fusion.out.weight", 1);
    k_bias<<<(3 * raw.H * raw.W + 255) / 256, 256>>>(raw.d, c.wt["fusion.out.bias"], d_out, 3, raw.H * raw.W);
    FR(s1.d); FR(s2.d); FR(s3.d); FR(s4.d); FR(u3.d); FR(c3.d); FR(d3.d); FR(u2.d); FR(c2.d); FR(d2.d); FR(u1.d); FR(c1.d); FR(d1.d); FR(raw.d);
}

}  // namespace

// In-pipeline entry point: run the fusion residual straight over device memory (e.g. a Vulkan-shared
// external buffer holding the packed 12-channel warp input), writing the [3,H,W] residual to d_out.
// Does NOT synchronize — the caller sequences it via the cross-API timeline semaphore.
void runFusionCUDADevice(const CnnModel& m, const void* d_in, void* d_out, int H, int W) {
    if (!m.valid) return;
    ensureInit();
    fusionCore(ensure(m), static_cast<const __half*>(d_in), static_cast<__half*>(d_out), H, W);
}

std::vector<float> runFusionCUDA(const CnnModel& m, const std::vector<float>& x, int H, int W) {
    if (!m.valid || (int)x.size() != 12 * H * W) return {};
    ensureInit();
    Cache& c = ensure(m);
    std::vector<__half> hx(x.size()); for (size_t i = 0; i < x.size(); ++i) hx[i] = __float2half(x[i]);
    T in = galloc(12, H, W); cudaMemcpy(in.d, hx.data(), sizeof(__half) * hx.size(), cudaMemcpyHostToDevice);
    T o = galloc(3, H, W);
    fusionCore(c, in.d, o.d, H, W);
    std::vector<__half> ho(3 * H * W); cudaMemcpy(ho.data(), o.d, sizeof(__half) * ho.size(), cudaMemcpyDeviceToHost);
    FR(in.d); FR(o.d);
    cudaStreamSynchronize(0);
    std::vector<float> out(ho.size()); for (size_t i = 0; i < out.size(); ++i) out[i] = __half2float(ho[i]);
    return out;
}

}  // namespace nvofg
