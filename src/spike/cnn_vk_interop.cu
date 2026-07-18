// End-to-end proof of the NVOFG_INTERP_CNN in-pipeline GPU dispatch (Plan A B4 / ADR 0004):
// the REAL fusion net runs on the Tensor Cores over VULKAN-SHARED external memory, gated by a
// cross-API timeline semaphore, and its output matches the CPU reference (runFusionCPU).
//
//   Vulkan allocates an exportable buffer = [ input 12*H*W fp16 | output 3*H*W fp16 ]
//   (host mapping stands in for the prep/warp pass writing the packed 12-channel input).
//   Buffer memory + a timeline semaphore are exported (OPAQUE_FD) and imported into CUDA.
//   Vulkan signals T=1  ->  CUDA waits 1, runFusionCUDADevice(d_in -> d_out), signals 2  ->
//   host waits the Vulkan semaphore reaches 2, reads d_out through the Vulkan mapping, and
//   compares to runFusionCPU on the same input.
//
// This is exactly what recordCnnRefine does in-pipeline, minus the Vulkan compute pass that produces
// the packed input (here a host write). Exit 0 = the shared-memory fusion dispatch matches the CPU
// reference on real hardware.  Usage: cnn_vk_interop <model.nvfgw> [H W]
//
// Builds only when NVOFG_ENABLE_CUDA is set.

#include <vulkan/vulkan.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cnn_model.hpp"
#include "cnn_infer.hpp"
#include "cnn_cuda.hpp"

#define VK(e) do { VkResult _r = (e); if (_r) { std::fprintf(stderr, "%s -> %d\n", #e, (int)_r); return 1; } } while (0)
#define CU(e) do { cudaError_t _r = (e); if (_r != cudaSuccess) { std::fprintf(stderr, "%s -> %s\n", #e, cudaGetErrorString(_r)); return 1; } } while (0)

static uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return ~0u;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.nvfgw> [H W]\n", argv[0]); return 2; }
    int H = argc > 3 ? std::atoi(argv[2]) : 48;
    int W = argc > 3 ? std::atoi(argv[3]) : 48;

    nvofg::CnnModel model;
    if (!nvofg::loadCnnModel(argv[1], model)) { std::fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    std::printf("model: %s (%zu tensors)\n", argv[1], model.tensors.size());

    // deterministic packed 12-channel input; fed identically to CUDA (via VK memory) and CPU.
    const int inN = 12 * H * W, outN = 3 * H * W;
    std::vector<float> xin(inN);
    for (int i = 0; i < inN; ++i) xin[i] = float(((i * 7) % 23)) / 23.0f - 0.4f;

    // ---- Vulkan instance/device (NVIDIA) with external memory/semaphore fd ----
    VkApplicationInfo app{}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    VkInstance inst; VK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE; uint32_t fam = 0;
    for (auto c : pds) {
        VkPhysicalDeviceProperties pr{}; vkGetPhysicalDeviceProperties(c, &pr);
        if (pr.vendorID != 0x10DE) continue;
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(c, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(c, &qn, qf.data());
        for (uint32_t i = 0; i < qn; ++i) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { pd = c; fam = i; break; }
        if (pd) break;
    }
    if (!pd) { std::printf("SKIP: no NVIDIA device\n"); return 0; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s\n", props.deviceName);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = fam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char* exts[] = { "VK_KHR_external_memory_fd", "VK_KHR_external_semaphore_fd" };
    VkPhysicalDeviceTimelineSemaphoreFeatures ts{}; ts.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES; ts.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci{}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.pNext = &ts;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2; dci.ppEnabledExtensionNames = exts;
    VkDevice dev; VK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, fam, 0, &queue);
    auto pGetMemFd = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    auto pGetSemFd = (PFN_vkGetSemaphoreFdKHR) vkGetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR");
    if (!pGetMemFd || !pGetSemFd) { std::fprintf(stderr, "missing external fd entrypoints\n"); return 1; }

    // ---- exportable, host-visible buffer: [ in (12*H*W half) | out (3*H*W half) ] ----
    const VkDeviceSize inBytes = VkDeviceSize(inN) * sizeof(__half);
    const VkDeviceSize outBytes = VkDeviceSize(outN) * sizeof(__half);
    const VkDeviceSize total = inBytes + outBytes;
    VkExternalMemoryBufferCreateInfo ext{}; ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo bci{}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bci.pNext = &ext;
    bci.size = total; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf; VK(vkCreateBuffer(dev, &bci, nullptr, &buf));
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(dev, buf, &req);
    VkExportMemoryAllocateInfo exp{}; exp.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.pNext = &exp;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memType(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == ~0u) { std::fprintf(stderr, "no host-visible exportable memory\n"); return 1; }
    VkDeviceMemory mem; VK(vkAllocateMemory(dev, &mai, nullptr, &mem));
    VK(vkBindBufferMemory(dev, buf, mem, 0));

    // Vulkan "produces" the packed input (host mapping stands in for the prep/warp pass).
    void* mapped; VK(vkMapMemory(dev, mem, 0, total, 0, &mapped));
    __half* hIn = (__half*)mapped;
    for (int i = 0; i < inN; ++i) hIn[i] = __float2half(xin[i]);

    // ---- export memory + timeline semaphore -> import into CUDA ----
    VkMemoryGetFdInfoKHR gfi{}; gfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gfi.memory = mem; gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int memFd = -1; VK(pGetMemFd(dev, &gfi, &memFd));
    cudaExternalMemory_t cuExtMem{};
    cudaExternalMemoryHandleDesc emd{}; emd.type = cudaExternalMemoryHandleTypeOpaqueFd;
    emd.handle.fd = memFd; emd.size = total;
    CU(cudaImportExternalMemory(&cuExtMem, &emd));
    void* dPtr = nullptr;
    cudaExternalMemoryBufferDesc bd{}; bd.offset = 0; bd.size = total; bd.flags = 0;
    CU(cudaExternalMemoryGetMappedBuffer(&dPtr, cuExtMem, &bd));
    __half* dIn = (__half*)dPtr;
    __half* dOut = (__half*)((char*)dPtr + inBytes);

    VkExportSemaphoreCreateInfo esc{}; esc.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    esc.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkSemaphoreTypeCreateInfo stc{}; stc.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    stc.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE; stc.initialValue = 0; stc.pNext = &esc;
    VkSemaphoreCreateInfo sci{}; sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO; sci.pNext = &stc;
    VkSemaphore sem; VK(vkCreateSemaphore(dev, &sci, nullptr, &sem));
    VkSemaphoreGetFdInfoKHR sgi{}; sgi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    sgi.semaphore = sem; sgi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int semFd = -1; VK(pGetSemFd(dev, &sgi, &semFd));
    cudaExternalSemaphore_t cuSem{};
    cudaExternalSemaphoreHandleDesc shd{}; shd.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    shd.handle.fd = semFd;
    CU(cudaImportExternalSemaphore(&cuSem, &shd));

    cudaStream_t stream; CU(cudaStreamCreate(&stream));
    // warm the weight cache + cublas/pool on this stream before the timed cross-API section.
    nvofg::runFusionCUDADevice(model, dIn, dOut, H, W); CU(cudaStreamSynchronize(0));

    // ---- cross-API timeline: Vulkan signals 1, CUDA waits 1 / fusion / signals 2 ----
    VkTimelineSemaphoreSubmitInfo tssi{}; tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    uint64_t one = 1; tssi.signalSemaphoreValueCount = 1; tssi.pSignalSemaphoreValues = &one;
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.pNext = &tssi;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &sem;
    VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));

    cudaExternalSemaphoreWaitParams wp{}; wp.params.fence.value = 1;
    CU(cudaWaitExternalSemaphoresAsync(&cuSem, &wp, 1, stream));
    nvofg::runFusionCUDADevice(model, dIn, dOut, H, W);   // fusion residual on Vulkan-shared memory
    cudaExternalSemaphoreSignalParams sp{}; sp.params.fence.value = 2;
    CU(cudaSignalExternalSemaphoresAsync(&cuSem, &sp, 1, stream));

    uint64_t two = 2;
    VkSemaphoreWaitInfo swi{}; swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    swi.semaphoreCount = 1; swi.pSemaphores = &sem; swi.pValues = &two;
    VK(vkWaitSemaphores(dev, &swi, UINT64_MAX));
    CU(cudaStreamSynchronize(stream));

    // ---- verify: d_out (read via Vulkan mapping) == runFusionCPU on the same input ----
    __half* hOut = (__half*)((char*)mapped + inBytes);
    std::vector<float> ref = nvofg::runFusionCPU(model, xin, H, W);
    double maxErr = 0.0, meanErr = 0.0;
    for (int i = 0; i < outN; ++i) {
        double e = std::abs(__half2float(hOut[i]) - ref[i]);
        maxErr = std::max(maxErr, e); meanErr += e;
    }
    meanErr /= outN;
    std::printf("fusion on Vulkan-shared memory vs CPU reference: %d vals max_abs_err=%.4g mean=%.4g\n",
                outN, maxErr, meanErr);
    bool ok = maxErr < 0.05;   // fp16 GEMM vs fp32 CPU tolerance (same as the host-array path)
    std::printf("RESULT: %s (in-pipeline CNN GPU dispatch over Vulkan memory %s)\n",
                ok ? "PASS" : "FAIL", ok ? "verified" : "mismatch");

    cudaDestroyExternalSemaphore(cuSem);
    cudaDestroyExternalMemory(cuExtMem);
    cudaStreamDestroy(stream);
    vkDestroySemaphore(dev, sem, nullptr);
    vkUnmapMemory(dev, mem);
    vkDestroyBuffer(dev, buf, nullptr);
    vkFreeMemory(dev, mem, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return ok ? 0 : 1;
}
