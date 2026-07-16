// Stage 1 — FORMAT PREP.
// Convert an app color frame to the single-channel luma the OFA consumes.
// Writes an R8_UNORM storage image that is (later) bound as an OFA INPUT image.
import common;

[[vk::binding(0, 0)]] Texture2D<float4> gColor;
[[vk::image_format("r8")]]
[[vk::binding(1, 0)]] RWTexture2D<float>  gLuma;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint w, h;
    gLuma.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    int2 p = int2(tid.xy);
    float3 c = gColor.Load(int3(p, 0)).rgb;
    gLuma[p] = luma709(c);
}
