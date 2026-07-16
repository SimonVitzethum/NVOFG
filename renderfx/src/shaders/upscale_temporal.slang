// RenderFX Temporal upscaling backend (TAAU, roadmap M3) — a vendor-neutral temporal
// anti-aliasing upscaler. Accumulates the (jittered) current frame into a reprojected
// history with neighborhood clamping to reject stale samples (anti-ghost). Uses the
// shared Frame Context (color/motion) + a RenderFX-owned history ping-pong.
[[vk::binding(0, 0)]] Texture2D<float4>   gCurrent;  // render-res color
[[vk::binding(1, 0)]] Texture2D<float4>   gHistory;  // previous accumulation (output res)
[[vk::binding(2, 0)]] Texture2D<float2>   gMotion;   // motion (output-res px); dummy if absent
[[vk::binding(3, 0)]] SamplerState        gLinear;
[[vk::binding(4, 0)]] RWTexture2D<float4> gDst;       // output (app present target)
[[vk::image_format("rgba16f")]]
[[vk::binding(5, 0)]] RWTexture2D<float4> gHistOut;   // new history

struct Push { uint dstW, dstH; float alpha; uint flags; };  // flags: bit0 reset, bit1 hasMotion
[[vk::push_constant]] Push pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.dstW || tid.y >= pc.dstH) return;
    float2 dim = float2(pc.dstW, pc.dstH);
    float2 uv = (float2(tid.xy) + 0.5) / dim;

    float3 cur = gCurrent.SampleLevel(gLinear, uv, 0).rgb;

    // 3x3 neighborhood min/max of the current frame -> the valid color box.
    float3 mn = cur, mx = cur;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            float3 s = gCurrent.SampleLevel(gLinear, uv + float2(dx, dy) / dim, 0).rgb;
            mn = min(mn, s); mx = max(mx, s);
        }

    float3 outc;
    if (pc.flags & 1u) {
        outc = cur;                          // reset / first frame
    } else {
        float2 prevUV = uv;
        if (pc.flags & 2u) prevUV = uv - gMotion.SampleLevel(gLinear, uv, 0) / dim;
        bool valid = all(prevUV >= 0.0) && all(prevUV <= 1.0);
        float3 hist = clamp(gHistory.SampleLevel(gLinear, prevUV, 0).rgb, mn, mx);  // anti-ghost
        outc = valid ? lerp(hist, cur, pc.alpha) : cur;
    }

    gDst[int2(tid.xy)]     = float4(outc, 1.0);
    gHistOut[int2(tid.xy)] = float4(outc, 1.0);
}
