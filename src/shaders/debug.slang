// Debug visualisation — renders an internal buffer (flow / confidence / occlusion)
// into an app-provided target so integrators can inspect the pipeline
// (design testing section). Selected via nvofg_set_debug_view.
import common;

[[vk::binding(0, 0)]] Texture2D<float2>   gFlowFwd;
[[vk::binding(1, 0)]] Texture2D<float2>   gFlowBwd;
[[vk::binding(2, 0)]] Texture2D<float>    gConfidence;
[[vk::binding(3, 0)]] Texture2D<float>    gOcclusion;
[[vk::binding(4, 0)]] RWTexture2D<float4> gOut;

struct Push { uint width, height, mode; };  // mode = NvofgDebugView
[[vk::push_constant]] Push pc;

// NvofgDebugView values
static const uint DBG_FLOW_FWD = 1u, DBG_FLOW_BWD = 2u, DBG_CONFIDENCE = 3u,
                  DBG_OCCLUSION = 4u, DBG_DISOCCLUSION = 5u;

float3 flowColor(float2 f) {
    // Middlebury-ish: direction -> hue via a simple wheel, magnitude -> saturation.
    float mag = saturate(length(f) / 32.0);
    float ang = atan2(f.y, f.x);                 // -pi..pi
    float h = (ang / 6.2831853) + 0.5;           // 0..1
    float3 rgb = saturate(abs(frac(h + float3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0);
    return lerp(float3(0.5, 0.5, 0.5), rgb, mag);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    int2 p = int2(tid.xy);
    float3 c;
    switch (pc.mode) {
        case DBG_FLOW_FWD:    c = flowColor(gFlowFwd[p]); break;
        case DBG_FLOW_BWD:    c = flowColor(gFlowBwd[p]); break;
        case DBG_CONFIDENCE:  c = float3(gConfidence[p]); break;
        case DBG_OCCLUSION:   c = float3(gOcclusion[p], 0.0, 0.0); break;   // red = occluded
        case DBG_DISOCCLUSION:c = float3(0.0, gOcclusion[p], 0.0); break;   // green (folded into occ in M2)
        default:              c = float3(0.0); break;
    }
    gOut[p] = float4(c, 1.0);
}
