// CUDA (Tensor-Core) FusionNet forward — the NVOFG_INTERP_CNN GPU backend (ADR 0004). Same result
// as runFusionCPU (verified against it), but on the GPU via im2col + cuBLAS fp16 GEMM (fp32 accum).
// Built only when NVOFG_ENABLE_CUDA is ON. The recordCnnRefine seam calls this over the warp output.
#pragma once
#include "cnn_model.hpp"
#include <vector>

namespace nvofg {

// Upload+cache the model weights on first call; run the fusion forward on x=[12,H,W]; return the
// residual [3*H*W]. Returns empty on error. (For the in-pipeline path the inputs/outputs are Vulkan
// images shared via external memory; this host-array entry point is the verifiable reference call.)
std::vector<float> runFusionCUDA(const CnnModel& m, const std::vector<float>& x, int H, int W);

// In-pipeline entry point (recordCnnRefine): run the fusion residual directly over DEVICE memory —
// d_in = packed [12,H,W] fp16 warp input, d_out = [3,H,W] fp16 residual. Both are device pointers the
// caller owns; for the Vulkan path they are external-memory buffers shared with CUDA. Enqueues on the
// current stream and does NOT synchronize — the caller gates it with the cross-API timeline semaphore
// (cudaWaitExternalSemaphoresAsync before, cudaSignalExternalSemaphoresAsync after). See
// src/spike/cnn_vk_interop.cu for the end-to-end Vulkan-shared verification on real hardware.
void runFusionCUDADevice(const CnnModel& m, const void* d_in, void* d_out, int H, int W);

}  // namespace nvofg
