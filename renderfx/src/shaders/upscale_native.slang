// RenderFX Native upscaling backend — a vendor-neutral bilinear upscale (design.md
// §16: the always-available baseline). Records into the app's command buffer; RenderFX
// owns no graph. For each destination pixel it linearly samples the source at the
// matching UV. (Lanczos/Catmull-Rom is a later quality step; bilinear is the baseline.)
[[vk::binding(0, 0)]] Texture2D<float4>   gSrc;
[[vk::binding(1, 0)]] SamplerState        gLinear;
[[vk::binding(2, 0)]] RWTexture2D<float4> gDst;

struct Push { uint dstW, dstH; };
[[vk::push_constant]] Push pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.dstW || tid.y >= pc.dstH) return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.dstW, pc.dstH);
    gDst[int2(tid.xy)] = gSrc.SampleLevel(gLinear, uv, 0);
}
