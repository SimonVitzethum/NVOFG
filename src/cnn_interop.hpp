// NVOFG_INTERP_CNN Vulkan<->CUDA interop (Plan A B4). Imports nvofg's exportable CNN in/out buffers
// and the pipeline timeline semaphore into CUDA once, then runs the Tensor-Core fusion as another
// stage on the SAME timeline: it waits the value the pack pass signalled, runs the fusion over the
// shared buffers, and signals the next value the residual-add pass waits on. C++-only interface
// (no CUDA/Vulkan types) so pipeline.cpp can call it without CUDA. Built only with NVOFG_ENABLE_CUDA.
#pragma once
#include <cstdint>

struct NvofgContext;

namespace nvofg {

// Create the interop (export+import the ctx->cnnInBuf/cnnOutBuf memory and ctx->timeline). Returns an
// opaque handle, or nullptr if unavailable (no model, missing fd entrypoints, import failure).
void* cnnInteropCreate(NvofgContext* ctx);

// Enqueue on the CUDA side: wait `waitVal` on the shared timeline, run the fusion over the shared
// buffers, signal `waitVal+1`. Returns the signalled value (the residual-add submit waits on it).
uint64_t cnnInteropDispatch(NvofgContext* ctx, void* handle, uint64_t waitVal);

void cnnInteropDestroy(void* handle);

}  // namespace nvofg
