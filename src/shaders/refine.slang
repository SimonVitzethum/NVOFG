// Stage 3 — FLOW REFINE.
// Upsample the coarse OFA flow grids to full resolution (bilinear), decode the
// S10.5 vectors, derive a continuous confidence from the cost map (req.#2), and —
// when bidirectional flow is available — a forward/backward consistency occlusion
// mask (req.#1). Flow/cost arrive as ByteAddressBuffers copied from the OFA images.
import common;

[[vk::binding(0, 0)]] ByteAddressBuffer gFlowFwd;   // gridW*gridH packed S10.5
[[vk::binding(1, 0)]] ByteAddressBuffer gCost;      // gridW*gridH bytes (R8_UINT)
[[vk::binding(2, 0)]] ByteAddressBuffer gFlowBwd;   // backward flow (bidir); else placeholder
[[vk::image_format("rg16f")]]
[[vk::binding(3, 0)]] RWTexture2D<float2> gRefinedFwd;
[[vk::image_format("rg16f")]]
[[vk::binding(4, 0)]] RWTexture2D<float2> gRefinedBwd;
[[vk::image_format("r8")]]
[[vk::binding(5, 0)]] RWTexture2D<float>  gConfidence;
[[vk::image_format("r8")]]
[[vk::binding(6, 0)]] RWTexture2D<float>  gOcclusion;
[[vk::binding(7, 0)]] Texture2D<float2>   gMotion;    // app motion vectors, px (prev->curr)
[[vk::binding(8, 0)]] Texture2D<float>    gDepth;     // curr depth (linear or device)

struct Push {
    uint  width, height, gridW, gridH, gridSize, flags;   // bit0 cost,1 bidir,2 motion,3 depth
    float sigma;
};
[[vk::push_constant]] Push pc;

static const uint FLAG_COST = 1u, FLAG_BIDIR = 2u, FLAG_MOTION = 4u, FLAG_DEPTH = 8u;

int2 clampCell(int2 c) { return clamp(c, int2(0, 0), int2(int(pc.gridW) - 1, int(pc.gridH) - 1)); }

float2 flowCell(ByteAddressBuffer buf, int2 cell) {
    cell = clampCell(cell);
    uint idx = uint(cell.y) * pc.gridW + uint(cell.x);
    return decodeS10_5(buf.Load(idx * 4));
}

// bilinear upsample of a grid flow field at pixel p
float2 upsample(ByteAddressBuffer buf, int2 p) {
    float2 g = (float2(p) + 0.5) / float(pc.gridSize) - 0.5;
    int2 g0 = int2(floor(g)); float2 f = g - float2(g0);
    float2 a = lerp(flowCell(buf, g0 + int2(0,0)), flowCell(buf, g0 + int2(1,0)), f.x);
    float2 b = lerp(flowCell(buf, g0 + int2(0,1)), flowCell(buf, g0 + int2(1,1)), f.x);
    return lerp(a, b, f.y);
}

// nearest-cell backward flow at an arbitrary pixel position
float2 flowBwdAt(float2 pos) {
    int2 cell = int2(round((pos + 0.5) / float(pc.gridSize) - 0.5));
    return flowCell(gFlowBwd, cell);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    int2 p = int2(tid.xy);

    float2 Ffwd = upsample(gFlowFwd, p);

    float cost = 0.0;
    if (pc.flags & FLAG_COST) {
        int2 c = clampCell(int2(round((float2(p) + 0.5) / float(pc.gridSize) - 0.5)));
        uint idx = uint(c.y) * pc.gridW + uint(c.x);
        uint word = gCost.Load(idx & ~3u);
        cost = float((word >> ((idx & 3u) * 8u)) & 0xffu);
    }
    float conf = confidenceFromCost(cost, pc.sigma);

    // Motion-vector fusion (req.#3): adaptively combine OFA flow with app MVs by
    // confidence — never fully replace either. Where the OFA is unreliable the
    // result leans on the engine MV; where they agree, confidence is boosted.
    if (pc.flags & FLAG_MOTION) {
        float2 Fmv = gMotion[p];
        Ffwd = lerp(Ffwd, Fmv, 1.0 - conf);
        float agree = 1.0 - saturate(length(Ffwd - Fmv) / 8.0);
        conf = saturate(max(conf, 0.5 * conf + 0.5 * agree));
    }
    gRefinedFwd[p] = Ffwd;

    float occ = 0.0;
    float2 Fbwd = float2(0.0);
    if (pc.flags & FLAG_BIDIR) {
        Fbwd = upsample(gFlowBwd, p);
        // forward-backward consistency: follow Ffwd, read backward flow there.
        float2 dst = float2(p) + Ffwd;
        float2 back = flowBwdAt(dst);
        float residual = length(Ffwd + back);               // ~0 if consistent
        float norm = max(length(Ffwd), 1.0);
        occ = saturate(residual / (0.5 * norm + 2.0));       // relative threshold
        conf *= (1.0 - 0.5 * occ);                           // occlusion lowers confidence
    }
    // Depth disocclusion (req.#4): strong depth discontinuities mark object
    // silhouettes, where newly-revealed background disoccludes — raise occlusion so
    // the warp fills from the non-occluded side instead of blending across the edge.
    if (pc.flags & FLAG_DEPTH) {
        float dC = gDepth[p];
        float dR = gDepth[int2(min(p.x + 1, int(pc.width) - 1), p.y)];
        float dL = gDepth[int2(max(p.x - 1, 0), p.y)];
        float dD = gDepth[int2(p.x, min(p.y + 1, int(pc.height) - 1))];
        float dU = gDepth[int2(p.x, max(p.y - 1, 0))];
        float grad = abs(dR - dL) + abs(dD - dU);
        float edge = saturate(grad / (abs(dC) * 0.1 + 1e-3));  // relative depth step
        occ = max(occ, edge);
    }
    gRefinedBwd[p] = Fbwd;
    gConfidence[p] = conf;
    gOcclusion[p]  = occ;
}
