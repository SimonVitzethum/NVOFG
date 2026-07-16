// Stage 4 — WARP + BLEND (WarpInterpolator, the v1 Interpolator implementation).
// Two-sided backward warp of prev/curr into the halfway frame, blended by the
// confidence from REFINE. Low-confidence pixels fall back to the temporally
// nearer real frame (anti-ghost). Occlusion-aware two-sided fill lands in M2.
import common;

[[vk::binding(0, 0)]] Texture2D<float4>  gPrev;
[[vk::binding(1, 0)]] Texture2D<float4>  gCurr;
[[vk::binding(2, 0)]] Texture2D<float2>  gFlow;        // full-res forward flow, px
[[vk::binding(3, 0)]] Texture2D<float>   gConfidence;  // full-res, [0,1]
[[vk::binding(4, 0)]] SamplerState       gLinear;
[[vk::binding(5, 0)]] RWTexture2D<float4> gOut;        // registered output (any color fmt)

struct Push {
    uint  width, height;
    float phase;       // t in (0,1); 0.5 = 2x
    uint  uiMasked;    // reserved; UI composite handled separately in M2
};
[[vk::push_constant]] Push pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    int2  p   = int2(tid.xy);
    float2 dim = float2(pc.width, pc.height);
    float  t  = pc.phase;

    float2 F    = gFlow[p];               // forward flow N-1 -> N at this position
    float  conf = gConfidence[p];

    // Backward-sample both real frames toward the midpoint.
    float2 prevUV = (float2(p) - t * F + 0.5) / dim;
    float2 currUV = (float2(p) + (1.0 - t) * F + 0.5) / dim;
    float4 prevC  = gPrev.SampleLevel(gLinear, prevUV, 0);
    float4 currC  = gCurr.SampleLevel(gLinear, currUV, 0);

    // Interpolated estimate.
    float4 interp = lerp(prevC, currC, t);

    // Anti-ghost: where confidence is low, prefer the nearer real frame sampled
    // in place (no warp) rather than a smeared blend.
    float4 nearest = (t < 0.5) ? gPrev.SampleLevel(gLinear, (float2(p) + 0.5) / dim, 0)
                               : gCurr.SampleLevel(gLinear, (float2(p) + 0.5) / dim, 0);
    float4 outC = lerp(nearest, interp, saturate(conf));

    gOut[p] = outC;
}
