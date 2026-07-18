#include "cnn_interop.hpp"
#include "internal.hpp"
#include "cnn_model.hpp"
#include "cnn_cuda.hpp"

#include <cuda_runtime.h>
#include <cstdio>

namespace nvofg {
namespace {

struct Interop {
    cudaExternalMemory_t emIn = nullptr, emOut = nullptr;
    cudaExternalSemaphore_t sem = nullptr;
    float* dIn = nullptr;
    float* dOut = nullptr;
    int H = 0, W = 0;
};

bool importBuffer(VkDevice dev, PFN_vkGetMemoryFdKHR pMemFd, const Buffer& b,
                  cudaExternalMemory_t& em, float** dptr) {
    VkMemoryGetFdInfoKHR gi{}; gi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gi.memory = b.memory; gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    if (pMemFd(dev, &gi, &fd) != VK_SUCCESS) return false;
    cudaExternalMemoryHandleDesc hd{}; hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
    hd.handle.fd = fd; hd.size = b.size;
    if (cudaImportExternalMemory(&em, &hd) != cudaSuccess) return false;
    cudaExternalMemoryBufferDesc bd{}; bd.offset = 0; bd.size = b.size; bd.flags = 0;
    return cudaExternalMemoryGetMappedBuffer((void**)dptr, em, &bd) == cudaSuccess;
}

}  // namespace

void* cnnInteropCreate(NvofgContext* ctx) {
    if (!ctx || !ctx->cnn || !ctx->cnnInBuf.buffer || !ctx->cnnOutBuf.buffer || !ctx->timeline)
        return nullptr;
    auto pMemFd = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(ctx->device, "vkGetMemoryFdKHR");
    auto pSemFd = (PFN_vkGetSemaphoreFdKHR) vkGetDeviceProcAddr(ctx->device, "vkGetSemaphoreFdKHR");
    if (!pMemFd || !pSemFd) return nullptr;

    Interop* it = new Interop();
    it->H = (int)ctx->height; it->W = (int)ctx->width;
    if (!importBuffer(ctx->device, pMemFd, ctx->cnnInBuf, it->emIn, &it->dIn) ||
        !importBuffer(ctx->device, pMemFd, ctx->cnnOutBuf, it->emOut, &it->dOut)) {
        cnnInteropDestroy(it); return nullptr;
    }
    // import the pipeline timeline semaphore (exported at creation when built with CUDA)
    VkSemaphoreGetFdInfoKHR sgi{}; sgi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    sgi.semaphore = ctx->timeline; sgi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int semFd = -1;
    if (pSemFd(ctx->device, &sgi, &semFd) != VK_SUCCESS) { cnnInteropDestroy(it); return nullptr; }
    cudaExternalSemaphoreHandleDesc shd{};
    shd.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd; shd.handle.fd = semFd;
    if (cudaImportExternalSemaphore(&it->sem, &shd) != cudaSuccess) { cnnInteropDestroy(it); return nullptr; }
    return it;
}

uint64_t cnnInteropDispatch(NvofgContext* ctx, void* handle, uint64_t waitVal) {
    Interop* it = (Interop*)handle;
    if (!it || !ctx->cnn) return waitVal;
    cudaExternalSemaphoreWaitParams wp{}; wp.params.fence.value = waitVal;
    cudaWaitExternalSemaphoresAsync(&it->sem, &wp, 1, 0);
    runFusionCUDADeviceF32(*ctx->cnn, it->dIn, it->dOut, it->H, it->W);   // Tensor-Core fusion
    cudaExternalSemaphoreSignalParams sp{}; sp.params.fence.value = waitVal + 1;
    cudaSignalExternalSemaphoresAsync(&it->sem, &sp, 1, 0);
    return waitVal + 1;
}

void cnnInteropDestroy(void* handle) {
    Interop* it = (Interop*)handle;
    if (!it) return;
    if (it->sem) cudaDestroyExternalSemaphore(it->sem);
    if (it->emIn) cudaDestroyExternalMemory(it->emIn);
    if (it->emOut) cudaDestroyExternalMemory(it->emOut);
    delete it;
}

}  // namespace nvofg
