// RenderFX RR/GBuffer debug visualisation. Renders one Frame Context channel for
// validating Ray Reconstruction inputs. A single sampled source is interpreted per
// `mode`; when the channel is absent (`present == 0`) the output is magenta — an
// immediate missing-input diagnostic. Vendor-neutral compute; runs on any GPU.
[[vk::binding(0, 0)]] Texture2D<float4>   gSrc;
[[vk::binding(1, 0)]] SamplerState        gLinear;
[[vk::binding(2, 0)]] RWTexture2D<float4> gOut;

struct Push { uint w, h, mode, present; };  // mode = RfxDebugView
[[vk::push_constant]] Push pc;

// keep in sync with RfxDebugView
static const uint V_NORMALS = 1u, V_ROUGHNESS = 2u, V_DIFFUSE = 3u, V_SPECULAR = 4u,
                  V_DEPTH = 5u, V_MOTION = 6u;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.w || tid.y >= pc.h) return;
    int2 p = int2(tid.xy);
    if (pc.present == 0u) { gOut[p] = float4(1.0, 0.0, 1.0, 1.0); return; }  // missing input

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.w, pc.h);
    float4 s = gSrc.SampleLevel(gLinear, uv, 0);
    float3 c;
    switch (pc.mode) {
        case V_NORMALS:  c = s.rgb * 0.5 + 0.5; break;       // [-1,1] -> [0,1]
        case V_ROUGHNESS:c = s.rrr; break;
        case V_DIFFUSE:
        case V_SPECULAR: c = s.rgb; break;
        case V_DEPTH:    c = saturate(s.rrr); break;
        case V_MOTION:   c = float3(s.rg * 0.5 + 0.5, 0.5); break;
        default:         c = float3(0.0); break;
    }
    gOut[p] = float4(c, 1.0);
}
