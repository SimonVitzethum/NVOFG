// Motion-vector HINT encode (OFA pre-pass, req.#3 "fuse in two places").
// Converts the app's full-resolution motion vectors into the OFA's S10.5 hint
// grid, packed into a buffer that is then copied into the SFIXED5 hint image the
// OFA reads. Feeding a hint steers the hardware search — better initial flow on
// low-texture / repetitive regions than pure OFA.

[[vk::binding(0, 0)]] Texture2D<float2>       gMotion;   // full-res app MVs, px (prev->curr)
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gHint;     // gridW*gridH packed S10.5

struct Push { uint gridW, gridH, gridSize, width, height; };
[[vk::push_constant]] Push pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.gridW || tid.y >= pc.gridH) return;
    int2 px = int2(tid.xy) * int(pc.gridSize) + int(pc.gridSize / 2);
    px = min(px, int2(int(pc.width) - 1, int(pc.height) - 1));
    float2 mv = gMotion[px];
    int xi = clamp(int(round(mv.x * 32.0)), -32768, 32767);   // S10.5
    int yi = clamp(int(round(mv.y * 32.0)), -32768, 32767);
    uint packed = (uint(xi) & 0xffffu) | ((uint(yi) & 0xffffu) << 16);
    gHint[tid.y * pc.gridW + tid.x] = packed;
}
