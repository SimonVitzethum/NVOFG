// Tier B — portable shader optical flow (design §8).
// A dense block-matching search on the luma pair, producing the same grid flow +
// cost layout the OFA path feeds into REFINE, so any Vulkan GPU (AMD/Intel) can run
// frame generation without the NVIDIA Optical Flow Accelerator. Lower quality and
// heavier than the OFA; a vendor-neutral fallback, not a replacement.
//
// One thread per grid cell. For the cell centre it searches offsets in
// [-searchR, searchR]^2 minimising the SAD of a (2*blockR+1)^2 luma block between
// prev and curr; the best offset is the forward flow, the mean abs diff is the cost.

[[vk::binding(0, 0)]] Texture2D<float>        gPrev;   // R8 luma (frame N-1)
[[vk::binding(1, 0)]] Texture2D<float>        gCurr;   // R8 luma (frame N)
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> gFlow;  // gridW*gridH packed S10.5
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> gCost;  // gridW*gridH cost (0..255)

struct Push {
    uint width, height, gridW, gridH, gridSize, searchR, blockR;
};
[[vk::push_constant]] Push pc;

float lum(Texture2D<float> t, int2 p, int2 dim) {
    p = clamp(p, int2(0, 0), dim - 1);
    return t[p];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.gridW || tid.y >= pc.gridH) return;
    int2 dim = int2(int(pc.width), int(pc.height));
    int2 c = int2(tid.xy) * int(pc.gridSize) + int(pc.gridSize) / 2;
    c = min(c, dim - 1);

    int R = int(pc.searchR);
    int B = int(pc.blockR);
    float best = 1e30;
    int2 bestOff = int2(0, 0);

    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            float sad = 0.0;
            for (int by = -B; by <= B; ++by) {
                for (int bx = -B; bx <= B; ++bx) {
                    float a = lum(gPrev, c + int2(bx, by), dim);
                    float b = lum(gCurr, c + int2(bx + dx, by + dy), dim);
                    sad += abs(a - b);
                }
            }
            if (sad < best) {
                best = sad;
                bestOff = int2(dx, dy);
            }
        }
    }

    uint idx = tid.y * pc.gridW + tid.x;
    int xi = clamp(bestOff.x * 32, -32768, 32767);   // S10.5
    int yi = clamp(bestOff.y * 32, -32768, 32767);
    gFlow[idx] = (uint(xi) & 0xffffu) | ((uint(yi) & 0xffffu) << 16);

    // Mean abs luma diff of the winning block -> cost (higher = less confident),
    // matching the OFA cost convention consumed by REFINE (confidence = exp(-cost/σ)).
    float area = float((2 * B + 1) * (2 * B + 1));
    float meanDiff = best / max(area, 1.0);          // 0..1
    gCost[idx] = uint(clamp(meanDiff * 4.0, 0.0, 1.0) * 255.0);
}
