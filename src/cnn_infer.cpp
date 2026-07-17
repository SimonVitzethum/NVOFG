#include "cnn_infer.hpp"
#include <cmath>
#include <cstdint>
#include <string>

namespace nvofg {
namespace {

float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) { if (m == 0) f = s; else { e = 113; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | (e << 23) | (m << 13); } }
    else if (e == 0x1F) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e - 15 + 127) << 23) | (m << 13);
    float o; __builtin_memcpy(&o, &f, 4); return o;
}

struct FT { std::vector<int> dims; std::vector<float> d; };  // a tensor as float

struct Ctx {
    const CnnModel& m;
    std::vector<float> W;  // scratch not used; tensors fetched on demand
    explicit Ctx(const CnnModel& mm) : m(mm) {}
    FT get(const std::string& n) const {
        FT t; auto it = m.tensors.find(n);
        if (it == m.tensors.end()) return t;
        t.dims = it->second.dims;
        t.d.resize(it->second.half.size());
        for (size_t i = 0; i < t.d.size(); ++i) t.d[i] = h2f(it->second.half[i]);
        return t;
    }
};

struct T { std::vector<float> d; int C, H, W; };

// conv2d cross-correlation, padding 1, kernel 3.
T conv3(const T& in, const FT& w, const FT& b, int stride) {
    int Ci = in.C, H = in.H, W = in.W, Co = w.dims[0];
    int Ho = (H + 2 - 3) / stride + 1, Wo = (W + 2 - 3) / stride + 1;
    T o; o.C = Co; o.H = Ho; o.W = Wo; o.d.assign((size_t)Co * Ho * Wo, 0.f);
    for (int co = 0; co < Co; ++co)
        for (int oy = 0; oy < Ho; ++oy)
            for (int ox = 0; ox < Wo; ++ox) {
                float s = b.d[co];
                for (int ci = 0; ci < Ci; ++ci)
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            int iy = oy * stride - 1 + ky, ix = ox * stride - 1 + kx;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                s += in.d[((size_t)ci * H + iy) * W + ix] * w.d[(((size_t)co * Ci + ci) * 3 + ky) * 3 + kx];
                        }
                o.d[((size_t)co * Ho + oy) * Wo + ox] = s;
            }
    return o;
}
inline float leaky(float x) { return x > 0 ? x : 0.1f * x; }
inline float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

T gated(const T& in, const Ctx& c, const std::string& p, int stride) {
    T a = conv3(in, c.get(p + ".feat.weight"), c.get(p + ".feat.bias"), stride);
    T g = conv3(in, c.get(p + ".gate.weight"), c.get(p + ".gate.bias"), stride);
    for (size_t i = 0; i < a.d.size(); ++i) a.d[i] = leaky(a.d[i]) * sigm(g.d[i]);
    return a;
}
T seq2(const T& in, const Ctx& c, const std::string& e, int stride) {
    return gated(gated(in, c, e + ".0", stride), c, e + ".1", 1);
}
T up(const T& in, int Ho, int Wo) {  // bilinear, align_corners=False
    T o; o.C = in.C; o.H = Ho; o.W = Wo; o.d.assign((size_t)in.C * Ho * Wo, 0.f);
    int H = in.H, W = in.W;
    for (int oy = 0; oy < Ho; ++oy) {
        float iy = (oy + 0.5f) * H / Ho - 0.5f; int y0 = (int)std::floor(iy); float fy = iy - y0;
        int y0c = y0 < 0 ? 0 : (y0 > H - 1 ? H - 1 : y0), y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) > H - 1 ? H - 1 : (y0 + 1));
        for (int ox = 0; ox < Wo; ++ox) {
            float ix = (ox + 0.5f) * W / Wo - 0.5f; int x0 = (int)std::floor(ix); float fx = ix - x0;
            int x0c = x0 < 0 ? 0 : (x0 > W - 1 ? W - 1 : x0), x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) > W - 1 ? W - 1 : (x0 + 1));
            for (int cc = 0; cc < in.C; ++cc) {
                const float* pp = &in.d[(size_t)cc * H * W];
                o.d[((size_t)cc * Ho + oy) * Wo + ox] =
                    pp[y0c * W + x0c] * (1 - fx) * (1 - fy) + pp[y0c * W + x1c] * fx * (1 - fy) +
                    pp[y1c * W + x0c] * (1 - fx) * fy + pp[y1c * W + x1c] * fx * fy;
            }
        }
    }
    return o;
}
T cat(const T& a, const T& b) {
    T o; o.C = a.C + b.C; o.H = a.H; o.W = a.W; o.d = a.d; o.d.insert(o.d.end(), b.d.begin(), b.d.end()); return o;
}

}  // namespace

std::vector<float> runFusionCPU(const CnnModel& m, const std::vector<float>& x, int H, int W) {
    if (!m.valid || (int)x.size() != 12 * H * W) return {};
    Ctx c(m);
    T in; in.C = 12; in.H = H; in.W = W; in.d = x;
    T s1 = seq2(in, c, "fusion.e1", 1), s2 = seq2(s1, c, "fusion.e2", 2),
      s3 = seq2(s2, c, "fusion.e3", 2), s4 = seq2(s3, c, "fusion.e4", 2);
    T d3 = seq2(cat(up(s4, s3.H, s3.W), s3), c, "fusion.d3", 1);
    T d2 = seq2(cat(up(d3, s2.H, s2.W), s2), c, "fusion.d2", 1);
    T d1 = seq2(cat(up(d2, s1.H, s1.W), s1), c, "fusion.d1", 1);
    T out = conv3(d1, c.get("fusion.out.weight"), c.get("fusion.out.bias"), 1);
    return out.d;
}

}  // namespace nvofg
