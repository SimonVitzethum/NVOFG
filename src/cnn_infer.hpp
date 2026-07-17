// FusionNet forward (Plan A B4) — the correctness reference the GPU backend (coopmat/CUDA) mirrors.
// Input x = concat(cand, prev, curr, flow_fwd, t_channel) = [12,H,W]; output = residual [3,H,W] the
// recordCnnRefine seam adds to the classical warp. Verified against the PyTorch export.
#pragma once
#include "cnn_model.hpp"
#include <vector>

namespace nvofg {

// CPU reference forward. Returns the fusion residual [3*H*W] (channel-major), or empty on a bad model.
std::vector<float> runFusionCPU(const CnnModel& m, const std::vector<float>& x, int H, int W);

}  // namespace nvofg
