// Shared helpers for the nvofg compute shaders.
// Slang, compiled to SPIR-V (see cmake/Shaders.cmake).

// Rec.709 luma. OFA tracks a single-channel image; luma preserves structure.
float luma709(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// Decode one OFA flow vector. The OFA writes VK_FORMAT_R16G16_SFIXED5_NV:
// two signed 16-bit components with 5 fractional bits (S10.5). We transport the
// raw int16s in a ByteAddressBuffer to avoid needing a special image view.
// `packed` is the 32-bit word (low 16 = x, high 16 = y).
float2 decodeS10_5(uint packed) {
    int xi = int(packed & 0xffffu);
    int yi = int((packed >> 16) & 0xffffu);
    // sign-extend 16-bit
    if (xi >= 0x8000) xi -= 0x10000;
    if (yi >= 0x8000) yi -= 0x10000;
    return float2(float(xi), float(yi)) / 32.0; // 5 fractional bits
}

// Cost -> continuous confidence in [0,1] (ADR 0002 req.#2). exp falloff: smooth,
// monotonic, never hard-cuts. Lower OFA cost = higher confidence.
float confidenceFromCost(float cost, float sigma) {
    return exp(-cost / max(sigma, 1e-3));
}
