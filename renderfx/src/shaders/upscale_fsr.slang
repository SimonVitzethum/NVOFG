// RenderFX FSR upscaling backend — an AMD FSR1-style *edge-adaptive spatial upscaler*
// (EASU family): vendor-neutral, runs on any Vulkan GPU, no SDK. For each destination
// pixel it reconstructs from a 4x4 source neighbourhood with an anisotropic Lanczos-2
// kernel whose lobe is stretched *along* the local edge and tightened *across* it — the
// core EASU idea (directional, edge-preserving reconstruction) rather than isotropic
// bilinear/bicubic. Single pass; RCAS-style sharpening is a documented follow-up.
//
// This is an independent implementation of the EASU approach, not AMD's bit-exact kernel.
[[vk::binding(0, 0)]] Texture2D<float4>   gSrc;   // render-resolution color (sampled, fetched via Load)
[[vk::binding(1, 0)]] RWTexture2D<float4> gDst;   // present-resolution target (storage)

struct Push { uint srcW, srcH, dstW, dstH; };
[[vk::push_constant]] Push pc;

float luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

float3 tap(int x, int y) {
    x = clamp(x, 0, int(pc.srcW) - 1);
    y = clamp(y, 0, int(pc.srcH) - 1);
    return gSrc.Load(int3(x, y, 0)).rgb;
}

// Lanczos-2 window (a=2). Preserves linear gradients exactly (no overshoot on ramps);
// its small negative lobes sharpen edges.
float lanczos2(float x) {
    x = abs(x);
    if (x < 1e-4) return 1.0;
    if (x >= 2.0) return 0.0;
    float px = 3.14159265 * x;
    return (sin(px) * sin(px * 0.5)) / (px * px * 0.5);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.dstW || tid.y >= pc.dstH) return;
    float2 srcSz = float2(pc.srcW, pc.srcH);
    float2 dstSz = float2(pc.dstW, pc.dstH);
    float2 sp = (float2(tid.xy) + 0.5) * srcSz / dstSz - 0.5;   // dst -> src texel space
    int2 ip = int2(floor(sp));
    float2 fp = sp - float2(ip);

    // Local edge analysis: Sobel on 3x3 luma around the nearest source texel.
    float l00 = luma(tap(ip.x-1, ip.y-1)), l10 = luma(tap(ip.x, ip.y-1)), l20 = luma(tap(ip.x+1, ip.y-1));
    float l01 = luma(tap(ip.x-1, ip.y  )),                                l21 = luma(tap(ip.x+1, ip.y  ));
    float l02 = luma(tap(ip.x-1, ip.y+1)), l12 = luma(tap(ip.x, ip.y+1)), l22 = luma(tap(ip.x+1, ip.y+1));
    float gx = (l20 + 2.0*l21 + l22) - (l00 + 2.0*l01 + l02);
    float gy = (l02 + 2.0*l12 + l22) - (l00 + 2.0*l10 + l20);
    float gm = sqrt(gx*gx + gy*gy) + 1e-5;
    float2 edge = float2(-gy, gx) / gm;        // unit vector ALONG the edge
    float2 across = float2(edge.y, -edge.x);   // perpendicular (across the edge)
    float strength = saturate(gm * 2.0);       // 0 flat .. 1 strong edge

    // Anisotropic Lanczos-2 over the 4x4 neighbourhood. Along-edge lobe widened,
    // across-edge lobe tightened -> preserves edges, suppresses stair-stepping.
    float2 scale = lerp(float2(1.0, 1.0), float2(1.4, 0.7), strength);
    float3 acc = float3(0.0);
    float wsum = 0.0;
    // Deringing: track the min/max of the nearest 2x2 taps (EASU clamps the result to
    // this range so its negative lobes can sharpen but never overshoot -> no ringing).
    float3 lo = float3(1e9), hi = float3(-1e9);
    [unroll] for (int dy = -1; dy <= 2; ++dy)
    [unroll] for (int dx = -1; dx <= 2; ++dx) {
        float3 c = tap(ip.x + dx, ip.y + dy);
        float2 off = float2(dx, dy) - fp;      // tap offset from sample centre (src texels)
        float a = dot(off, edge) / scale.x;
        float b = dot(off, across) / scale.y;
        float w = lanczos2(a) * lanczos2(b);
        acc += c * w;
        wsum += w;
        if (dx >= 0 && dx <= 1 && dy >= 0 && dy <= 1) { lo = min(lo, c); hi = max(hi, c); }
    }
    float3 col = (wsum > 1e-5) ? acc / wsum : tap(ip.x, ip.y);
    col = clamp(col, lo, hi);               // suppress ringing (EASU deringing)
    gDst[int2(tid.xy)] = float4(max(col, float3(0.0)), 1.0);
}
