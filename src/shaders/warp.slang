// Stage 4 — WARP + BLEND (WarpInterpolator, the v1 Interpolator implementation).
// Two-sided backward warp of prev/curr into the halfway frame, blended by the
// confidence + occlusion from REFINE, with the mandatory UI mask and the reactive
// mask applied last (never interpolate HUD/particles).
import common;

[[vk::binding(0, 0)]] Texture2D<float4>  gPrev;
[[vk::binding(1, 0)]] Texture2D<float4>  gCurr;
[[vk::binding(2, 0)]] Texture2D<float2>  gFlowFwd;     // full-res forward flow, px
[[vk::binding(3, 0)]] Texture2D<float2>  gFlowBwd;     // full-res backward flow, px
[[vk::binding(4, 0)]] Texture2D<float>   gConfidence;  // [0,1]
[[vk::binding(5, 0)]] Texture2D<float>   gOcclusion;   // [0,1], 1 = fwd/bwd inconsistent
[[vk::binding(6, 0)]] Texture2D<float>   gUiMask;      // 1 = UI (never interpolate)
[[vk::binding(7, 0)]] Texture2D<float>   gReactive;    // 1 = reactive (suppress interp)
[[vk::binding(8, 0)]] SamplerState       gLinear;
[[vk::binding(9, 0)]] RWTexture2D<float4> gOut;
[[vk::binding(10, 0)]] Texture2D<float>  gMaterial;   // object/material id (optional)

struct Push {
    uint  width, height;
    float phase;       // t in (0,1); 0.5 = 2x
    uint  flags;       // bit0 hasUi,1 hasReactive,2 hasBidir,3 hasMaterial
};
[[vk::push_constant]] Push pc;

static const uint FLAG_UI = 1u, FLAG_REACTIVE = 2u, FLAG_BIDIR = 4u, FLAG_MATERIAL = 8u;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    int2   p   = int2(tid.xy);
    float2 dim = float2(pc.width, pc.height);
    float  t   = pc.phase;
    float2 uvP = (float2(p) + 0.5) / dim;

    float2 Ffwd = gFlowFwd[p];
    float  conf = gConfidence[p];
    float  occ  = (pc.flags & FLAG_BIDIR) ? gOcclusion[p] : 0.0;

    // Two-sided backward sampling toward the midpoint. The prev side follows the
    // forward flow; the curr side follows the backward flow when available (an
    // independent estimate that fills prev-side disocclusions), else forward flow.
    float2 prevUV = uvP - t * Ffwd / dim;
    float2 currUV;
    if (pc.flags & FLAG_BIDIR) {
        float2 Fbwd = gFlowBwd[p];   // N -> N-1; curr side advances by -(1-t)*Fbwd
        currUV = uvP - (1.0 - t) * Fbwd / dim;
    } else {
        currUV = uvP + (1.0 - t) * Ffwd / dim;
    }
    float4 prevC = gPrev.SampleLevel(gLinear, prevUV, 0);
    float4 currC = gCurr.SampleLevel(gLinear, currUV, 0);

    // Object/material boundaries: don't blend across different objects (req.#5).
    float matEdge = 0.0;
    if (pc.flags & FLAG_MATERIAL) {
        float m  = gMaterial[p];
        float mR = gMaterial[int2(min(p.x + 1, int(pc.width) - 1), p.y)];
        float mD = gMaterial[int2(p.x, min(p.y + 1, int(pc.height) - 1))];
        matEdge = (abs(mR - m) + abs(mD - m) > 1e-4) ? 1.0 : 0.0;
    }

    // Occlusion + material edges lower effective confidence (they ghost when blended).
    float eff = saturate(conf * (1.0 - max(occ, matEdge)));

    float4 interp  = lerp(prevC, currC, t);
    float4 nearest = (t < 0.5) ? gPrev.SampleLevel(gLinear, uvP, 0)
                               : gCurr.SampleLevel(gLinear, uvP, 0);
    float4 outC = lerp(nearest, interp, eff);

    // Reactive (alpha-test/particles/water): flow is meaningless -> nearest real.
    if ((pc.flags & FLAG_REACTIVE) && gReactive[p] > 0.5)
        outC = nearest;

    // UI mask is mandatory: HUD/crosshair/text must come straight from a real
    // frame, never interpolated. Take the most recent (curr).
    if ((pc.flags & FLAG_UI) && gUiMask[p] > 0.5)
        outC = gCurr.SampleLevel(gLinear, uvP, 0);

    gOut[p] = outC;
}
