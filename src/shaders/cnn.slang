// NVOFG_INTERP_CNN interop bridge (Plan A B4). Two compute passes that flank the CUDA Tensor-Core
// fusion (runFusionCUDADeviceF32) over Vulkan-shared buffers:
//   pack: gather the warp candidate + prev/curr/flow + phase into the packed [12,H,W] fp32 input
//         buffer the CNN reads (channel-major: cand.rgb, prev.rgb, curr.rgb, flow.xy, t).
//   add : add the CNN's [3,H,W] fp32 residual back onto the warp output image (residual-on-warp).
// fp32 buffers avoid needing 16-bit storage on the Vulkan side; CUDA converts f32<->fp16 internally.

struct Push { uint width, height; float phase; };
[[vk::push_constant]] Push pc;

// ---- pack -------------------------------------------------------------------------------------
[[vk::binding(0, 0)]] RWTexture2D<float4>      gCand;   // warp output = candidate (read)
[[vk::binding(1, 0)]] Texture2D<float4>        gPrev;
[[vk::binding(2, 0)]] Texture2D<float4>        gCurr;
[[vk::binding(3, 0)]] Texture2D<float2>        gFlow;   // full-res forward flow, px
[[vk::binding(4, 0)]] RWStructuredBuffer<float> gIn;    // [12*H*W] channel-major

void put(uint c, uint2 p, float v) { gIn[(c * pc.height + p.y) * pc.width + p.x] = v; }

[shader("compute")]
[numthreads(8, 8, 1)]
void pack(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    int2 p = int2(tid.xy);
    float4 cand = gCand[p];
    float4 prev = gPrev.Load(int3(p, 0));
    float4 curr = gCurr.Load(int3(p, 0));
    float2 fl   = gFlow.Load(int3(p, 0));
    uint2 u = uint2(tid.xy);
    put(0, u, cand.r); put(1, u, cand.g); put(2, u, cand.b);
    put(3, u, prev.r); put(4, u, prev.g); put(5, u, prev.b);
    put(6, u, curr.r); put(7, u, curr.g); put(8, u, curr.b);
    put(9, u, fl.x);   put(10, u, fl.y);  put(11, u, pc.phase);
}

// ---- add --------------------------------------------------------------------------------------
[[vk::binding(0, 0)]] RWTexture2D<float4>      gOut;    // warp output image (read-modify-write)
[[vk::binding(1, 0)]] RWStructuredBuffer<float> gRes;   // [3*H*W] channel-major residual

[shader("compute")]
[numthreads(8, 8, 1)]
void add(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    int2 p = int2(tid.xy);
    uint base = tid.y * pc.width + tid.x;
    uint plane = pc.width * pc.height;
    float3 res = float3(gRes[base], gRes[plane + base], gRes[2 * plane + base]);
    float4 c = gOut[p];
    c.rgb += res;
    gOut[p] = c;
}
