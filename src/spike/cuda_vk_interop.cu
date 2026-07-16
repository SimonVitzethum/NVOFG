// CUDA<->Vulkan interop feasibility spike (design analysis §D): prove the full
// pattern a CUDA Tensor-Core interpolator backend needs —
//   Vulkan-allocated memory  --(OPAQUE_FD)-->  CUDA device pointer
//   Vulkan timeline semaphore --(OPAQUE_FD)--> CUDA external semaphore
// so a CUDA WMMA kernel runs on Vulkan buffers, gated by a cross-API timeline:
//   Vulkan signals T=1  ->  CUDA waits 1, runs Tensor-Core GEMM, signals 2  ->
//   host waits Vulkan semaphore reaches 2 and verifies the result.
//
// Builds only when NVOFG_ENABLE_CUDA is set (needs the Vulkan loader + CUDA).
// Exit 0 = interop + Tensor-Core execution on shared memory verified.

#include <vulkan/vulkan.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace nvcuda;
static constexpr int M = 64, N = 64, K = 64, T = 16;

__global__ void wmma_gemm(const half* A, const half* B, float* C) {
    int warpM = blockIdx.x, warpN = blockIdx.y;
    wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, T, T, T, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int k = 0; k < K; k += T) {
        wmma::load_matrix_sync(a, A + (warpM * T) * K + k, K);
        wmma::load_matrix_sync(b, B + k * N + (warpN * T), N);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + (warpM * T) * N + (warpN * T), c, N, wmma::mem_row_major);
}

#define VK(e)                                                                   \
    do { VkResult _r = (e); if (_r) { std::fprintf(stderr, "%s -> %d\n", #e, (int)_r); return 1; } } while (0)
#define CU(e)                                                                   \
    do { cudaError_t _r = (e); if (_r != cudaSuccess) { std::fprintf(stderr, "%s -> %s\n", #e, cudaGetErrorString(_r)); return 1; } } while (0)

static uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return ~0u;
}

int main() {
    // ---- Vulkan instance/device (NVIDIA) with external memory/semaphore fd ----
    VkApplicationInfo app{}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    VkInstance inst; VK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE; uint32_t fam = 0;
    for (auto c : pds) {
        VkPhysicalDeviceProperties pr{}; vkGetPhysicalDeviceProperties(c, &pr);
        if (pr.vendorID != 0x10DE) continue;   // NVIDIA
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

    // ---- an exportable, host-visible buffer holding A|B (half) and C (float) ----
    const VkDeviceSize aBytes = VkDeviceSize(M) * K * sizeof(half);
    const VkDeviceSize bBytes = VkDeviceSize(K) * N * sizeof(half);
    const VkDeviceSize cBytes = VkDeviceSize(M) * N * sizeof(float);
    const VkDeviceSize total = aBytes + bBytes + cBytes;

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

    // Vulkan side "produces" A and B (host-visible mapping stands in for a Vulkan pass).
    void* mapped; VK(vkMapMemory(dev, mem, 0, total, 0, &mapped));
    half* hA = (half*)mapped;
    half* hB = (half*)((char*)mapped + aBytes);
    for (int i = 0; i < M * K; ++i) hA[i] = __float2half(float((i / K + i % K) % 3));
    for (int i = 0; i < K * N; ++i) hB[i] = __float2half(float((i / N + i % N) % 3));

    // ---- export memory fd -> import into CUDA ----
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
    half* dA = (half*)dPtr;
    half* dB = (half*)((char*)dPtr + aBytes);
    float* dC = (float*)((char*)dPtr + aBytes + bBytes);

    // ---- exportable timeline semaphore -> import into CUDA ----
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

    // ---- cross-API timeline: Vulkan signals 1, CUDA waits 1 / runs / signals 2 ----
    VkTimelineSemaphoreSubmitInfo tssi{}; tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    uint64_t one = 1; tssi.signalSemaphoreValueCount = 1; tssi.pSignalSemaphoreValues = &one;
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.pNext = &tssi;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &sem;
    VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));   // Vulkan "produced" A,B -> signal 1

    cudaExternalSemaphoreWaitParams wp{}; wp.params.fence.value = 1;
    CU(cudaWaitExternalSemaphoresAsync(&cuSem, &wp, 1, stream));
    dim3 grid(M / T, N / T);
    wmma_gemm<<<grid, 32, 0, stream>>>(dA, dB, dC);      // Tensor Cores on Vulkan memory
    cudaExternalSemaphoreSignalParams sp{}; sp.params.fence.value = 2;
    CU(cudaSignalExternalSemaphoresAsync(&cuSem, &sp, 1, stream));

    // Vulkan/host waits until CUDA's result is ready (timeline reaches 2).
    uint64_t two = 2;
    VkSemaphoreWaitInfo swi{}; swi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    swi.semaphoreCount = 1; swi.pSemaphores = &sem; swi.pValues = &two;
    VK(vkWaitSemaphores(dev, &swi, UINT64_MAX));
    CU(cudaStreamSynchronize(stream));

    // ---- verify: C = A*B, read through the Vulkan mapping ----
    float* hC = (float*)((char*)mapped + aBytes + bBytes);
    double maxErr = 0.0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += float((i + k) % 3) * float((k + j) % 3);
            maxErr = std::max(maxErr, (double)std::abs(hC[i * N + j] - acc));
        }
    std::printf("CUDA WMMA on Vulkan-shared memory: max abs error = %.3f\n", maxErr);

    bool ok = maxErr < 1e-3;
    std::printf("RESULT: %s (CUDA<->Vulkan interop + Tensor Cores %s)\n",
                ok ? "PASS" : "FAIL", ok ? "verified" : "mismatch");

    // teardown (CUDA owns the imported fds)
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
