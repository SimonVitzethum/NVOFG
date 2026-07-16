// Stage 3 — FLOW REFINE.
// Upsample the coarse OFA flow grid to full resolution (bilinear), decode the
// S10.5 vectors to pixel space, and derive a continuous confidence from the cost
// map (ADR 0002 req.#2). Bidirectional/MV/depth fusion attach here in M2.
//
// Flow and cost arrive as ByteAddressBuffers (copied from the OFA output images)
// so no special image views are needed.
import common;

[[vk::binding(0, 0)]] ByteAddressBuffer gFlow;   // gridW*gridH packed S10.5 words
[[vk::binding(1, 0)]] ByteAddressBuffer gCost;   // gridW*gridH bytes (R8_UINT)
[[vk::image_format("rg16f")]]
[[vk::binding(2, 0)]] RWTexture2D<float2> gRefinedFlow;  // full-res forward flow, px
[[vk::image_format("r8")]]
[[vk::binding(3, 0)]] RWTexture2D<float>  gConfidence;   // full-res, [0,1]

struct Push {
    uint  width, height;
    uint  gridW, gridH;
    uint  gridSize;
    uint  hasCost;
    float sigma;
};
[[vk::push_constant]] Push pc;

float2 flowAtCell(int2 cell) {
    cell = clamp(cell, int2(0, 0), int2(int(pc.gridW) - 1, int(pc.gridH) - 1));
    uint idx = uint(cell.y) * pc.gridW + uint(cell.x);
    return decodeS10_5(gFlow.Load(idx * 4));
}

float costAtCell(int2 cell) {
    if (pc.hasCost == 0) return 0.0;
    cell = clamp(cell, int2(0, 0), int2(int(pc.gridW) - 1, int(pc.gridH) - 1));
    uint idx  = uint(cell.y) * pc.gridW + uint(cell.x);
    uint word = gCost.Load((idx & ~3u));          // aligned 4-byte load
    return float((word >> ((idx & 3u) * 8u)) & 0xffu);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    int2 p = int2(tid.xy);

    // grid-space coordinate of this pixel's center
    float2 g = (float2(p) + 0.5) / float(pc.gridSize) - 0.5;
    int2   g0 = int2(floor(g));
    float2 f  = g - float2(g0);

    // bilinear flow upsample
    float2 f00 = flowAtCell(g0 + int2(0, 0));
    float2 f10 = flowAtCell(g0 + int2(1, 0));
    float2 f01 = flowAtCell(g0 + int2(0, 1));
    float2 f11 = flowAtCell(g0 + int2(1, 1));
    float2 flow = lerp(lerp(f00, f10, f.x), lerp(f01, f11, f.x), f.y);

    // confidence from nearest cost cell
    float cost = costAtCell(int2(round(g)));
    float conf = confidenceFromCost(cost, pc.sigma);

    gRefinedFlow[p] = flow;
    gConfidence[p]  = conf;
}
