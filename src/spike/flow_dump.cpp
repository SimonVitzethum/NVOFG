// nvofg M0 (full): run one real vkCmdOpticalFlowExecuteNV and dump the flow field.
//
// Proves the *native* OFA path end to end with zero use of the NvOF SDK: build a
// synthetic image pair with a known ground-truth motion, run the hardware optical
// flow, read the flow field back, decode the S10.5 vectors, and write a PPM
// visualisation plus a numeric sanity check against the ground truth.
//
// This is a one-shot offline tool: it favours clarity over the hot-path rules the
// library core follows (it uses vkQueueWaitIdle between steps, concurrent image
// sharing, etc.). The library proper does none of that.
//
// Usage: nvofg_flow_dump [out.ppm]   (default: flow.ppm in cwd)

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 256;
constexpr uint32_t kGrid = 4;               // output grid: one vector per 4x4 block
constexpr uint32_t kOutW = kWidth / kGrid;  // 64
constexpr uint32_t kOutH = kHeight / kGrid; // 64
constexpr int kShiftX = 8;                  // ground-truth motion: content moves +8 px in x
constexpr int kShiftY = 0;

#define CHECK(expr)                                                              \
    do {                                                                        \
        VkResult _r = (expr);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            std::fprintf(stderr, "%s failed: %d\n", #expr, (int)_r);            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

uint32_t findMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    std::fprintf(stderr, "no suitable memory type\n");
    std::exit(1);
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

Image createOfImage(VkPhysicalDevice pd, VkDevice dev, uint32_t w, uint32_t h,
                    VkFormat fmt, VkOpticalFlowUsageFlagsNV ofUsage,
                    VkImageUsageFlags usage, const uint32_t* families, uint32_t familyCount) {
    Image img;

    VkOpticalFlowImageFormatInfoNV ofInfo{};
    ofInfo.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV;
    ofInfo.usage = ofUsage;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.pNext = &ofInfo;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_CONCURRENT;
    ici.queueFamilyIndexCount = familyCount;
    ici.pQueueFamilyIndices = families;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CHECK(vkCreateImage(dev, &ici, nullptr, &img.image));

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(dev, img.image, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(vkAllocateMemory(dev, &mai, nullptr, &img.memory));
    CHECK(vkBindImageMemory(dev, img.image, img.memory, 0));

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CHECK(vkCreateImageView(dev, &vci, nullptr, &img.view));
    return img;
}

// Immediate one-shot command buffer on the given queue.
struct OneShot {
    VkDevice dev;
    VkCommandPool pool;
    VkQueue queue;
    VkCommandBuffer cmd;

    OneShot(VkDevice d, uint32_t family, VkQueue q) : dev(d), queue(q) {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = family;
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        CHECK(vkCreateCommandPool(dev, &pci, nullptr, &pool));
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        CHECK(vkAllocateCommandBuffers(dev, &ai, &cmd));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cmd, &bi));
    }
    void submit() {
        CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        CHECK(vkQueueWaitIdle(queue));
        vkDestroyCommandPool(dev, pool, nullptr);
    }
};

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
             VkAccessFlags srcA, VkAccessFlags dstA) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = srcA;
    b.dstAccessMask = dstA;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// A textured pattern so the OFA has features to track; shifted copy = known motion.
uint8_t pattern(int x, int y) {
    // Overlapping sinusoids + checker → high-frequency, non-repeating-ish texture.
    float v = 0.5f + 0.25f * std::sin(x * 0.20f) + 0.25f * std::sin(y * 0.17f);
    if (((x >> 4) ^ (y >> 4)) & 1) v = 1.0f - v;
    int iv = (int)(v * 255.0f);
    return (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
}

}  // namespace

int main(int argc, char** argv) {
    const char* outPath = (argc > 1) ? argv[1] : "flow.ppm";

    // ---- instance & device ----------------------------------------------
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t pdCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdCount, nullptr);
    std::vector<VkPhysicalDevice> pds(pdCount);
    vkEnumeratePhysicalDevices(instance, &pdCount, pds.data());

    VkPhysicalDevice pd = VK_NULL_HANDLE;
    uint32_t gfxFamily = ~0u, ofFamily = ~0u;
    for (VkPhysicalDevice cand : pds) {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(cand, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(cand, nullptr, &extCount, exts.data());
        bool ofExt = false;
        for (auto& e : exts)
            if (std::strcmp(e.extensionName, VK_NV_OPTICAL_FLOW_EXTENSION_NAME) == 0) ofExt = true;
        if (!ofExt) continue;

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &qfCount, qfs.data());
        uint32_t g = ~0u, o = ~0u;
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && g == ~0u) g = i;
            if ((qfs[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) && o == ~0u) o = i;
        }
        if (g != ~0u && o != ~0u) { pd = cand; gfxFamily = g; ofFamily = o; break; }
    }
    if (pd == VK_NULL_HANDLE) {
        std::fprintf(stderr, "no device with VK_NV_optical_flow + graphics+OF queues\n");
        return 2;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s  (gfx family %u, OF family %u)\n", props.deviceName, gfxFamily, ofFamily);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qcis[2]{};
    qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcis[0].queueFamilyIndex = gfxFamily;
    qcis[0].queueCount = 1;
    qcis[0].pQueuePriorities = &prio;
    qcis[1] = qcis[0];
    qcis[1].queueFamilyIndex = ofFamily;
    uint32_t qciCount = (gfxFamily == ofFamily) ? 1 : 2;

    VkPhysicalDeviceOpticalFlowFeaturesNV ofFeat{};
    ofFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
    ofFeat.opticalFlow = VK_TRUE;

    const char* devExts[] = {VK_NV_OPTICAL_FLOW_EXTENSION_NAME};
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &ofFeat;
    dci.queueCreateInfoCount = qciCount;
    dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = devExts;
    VkDevice dev = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));

    VkQueue gfxQueue, ofQueue;
    vkGetDeviceQueue(dev, gfxFamily, 0, &gfxQueue);
    vkGetDeviceQueue(dev, ofFamily, 0, &ofQueue);

    auto loadDev = [&](const char* n) { return vkGetDeviceProcAddr(dev, n); };
    auto pCreateSession = (PFN_vkCreateOpticalFlowSessionNV)loadDev("vkCreateOpticalFlowSessionNV");
    auto pDestroySession = (PFN_vkDestroyOpticalFlowSessionNV)loadDev("vkDestroyOpticalFlowSessionNV");
    auto pBindImage = (PFN_vkBindOpticalFlowSessionImageNV)loadDev("vkBindOpticalFlowSessionImageNV");
    auto pExecute = (PFN_vkCmdOpticalFlowExecuteNV)loadDev("vkCmdOpticalFlowExecuteNV");
    if (!pCreateSession || !pBindImage || !pExecute) {
        std::fprintf(stderr, "failed to load VK_NV_optical_flow entrypoints\n");
        return 1;
    }

    const uint32_t families[2] = {gfxFamily, ofFamily};
    const uint32_t famCount = (gfxFamily == ofFamily) ? 1 : 2;

    // ---- images ----------------------------------------------------------
    Image input = createOfImage(pd, dev, kWidth, kHeight, VK_FORMAT_R8_UNORM,
                                VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                families, famCount);
    Image reference = createOfImage(pd, dev, kWidth, kHeight, VK_FORMAT_R8_UNORM,
                                    VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                    families, famCount);
    Image flow = createOfImage(pd, dev, kOutW, kOutH, VK_FORMAT_R16G16_SFIXED5_NV,
                               VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                               families, famCount);

    // ---- staging upload of the synthetic pair ----------------------------
    auto uploadPattern = [&](Image& img, int shiftX, int shiftY) {
        std::vector<uint8_t> px(kWidth * kHeight);
        for (uint32_t y = 0; y < kHeight; ++y)
            for (uint32_t x = 0; x < kWidth; ++x)
                px[y * kWidth + x] = pattern((int)x - shiftX, (int)y - shiftY);

        VkBuffer buf;
        VkDeviceMemory mem;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = px.size();
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, buf, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemoryType(pd, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        CHECK(vkBindBufferMemory(dev, buf, mem, 0));
        void* mapped;
        CHECK(vkMapMemory(dev, mem, 0, px.size(), 0, &mapped));
        std::memcpy(mapped, px.data(), px.size());
        vkUnmapMemory(dev, mem);

        OneShot os(dev, gfxFamily, gfxQueue);
        barrier(os.cmd, img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {kWidth, kHeight, 1};
        vkCmdCopyBufferToImage(os.cmd, buf, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier(os.cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT);
        os.submit();
        vkDestroyBuffer(dev, buf, nullptr);
        vkFreeMemory(dev, mem, nullptr);
    };
    uploadPattern(input, 0, 0);
    uploadPattern(reference, kShiftX, kShiftY);  // content shifted → OFA should report (+8,0)

    // flow output image → GENERAL
    {
        OneShot os(dev, gfxFamily, gfxQueue);
        barrier(os.cmd, flow.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        os.submit();
    }

    // ---- optical flow session -------------------------------------------
    VkOpticalFlowSessionCreateInfoNV sci{};
    sci.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV;
    sci.width = kWidth;
    sci.height = kHeight;
    sci.imageFormat = VK_FORMAT_R8_UNORM;
    sci.flowVectorFormat = VK_FORMAT_R16G16_SFIXED5_NV;
    sci.costFormat = VK_FORMAT_UNDEFINED;
    sci.outputGridSize = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV;
    sci.hintGridSize = VK_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_NV;
    sci.performanceLevel = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_NV;  // best quality
    sci.flags = 0;
    VkOpticalFlowSessionNV session = VK_NULL_HANDLE;
    CHECK(pCreateSession(dev, &sci, nullptr, &session));

    CHECK(pBindImage(dev, session, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
                     input.view, VK_IMAGE_LAYOUT_GENERAL));
    CHECK(pBindImage(dev, session, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV,
                     reference.view, VK_IMAGE_LAYOUT_GENERAL));
    CHECK(pBindImage(dev, session, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV,
                     flow.view, VK_IMAGE_LAYOUT_GENERAL));

    // ---- execute on the optical-flow queue ------------------------------
    {
        OneShot os(dev, ofFamily, ofQueue);
        VkOpticalFlowExecuteInfoNV ei{};
        ei.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV;
        ei.flags = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV;  // single pair, no history
        pExecute(os.cmd, session, &ei);
        os.submit();
    }

    // ---- read the flow field back ---------------------------------------
    const VkDeviceSize flowBytes = (VkDeviceSize)kOutW * kOutH * 4;  // R16G16 = 4 bytes
    VkBuffer rb;
    VkDeviceMemory rbMem;
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = flowBytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CHECK(vkCreateBuffer(dev, &bci, nullptr, &rb));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, rb, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemoryType(pd, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CHECK(vkAllocateMemory(dev, &mai, nullptr, &rbMem));
        CHECK(vkBindBufferMemory(dev, rb, rbMem, 0));

        OneShot os(dev, gfxFamily, gfxQueue);
        barrier(os.cmd, flow.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {kOutW, kOutH, 1};
        vkCmdCopyImageToBuffer(os.cmd, flow.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &region);
        os.submit();
    }

    int16_t* fv;
    CHECK(vkMapMemory(dev, rbMem, 0, flowBytes, 0, (void**)&fv));

    // ---- decode S10.5, sanity-check, and write a PPM visualisation -------
    double sumX = 0, sumY = 0;
    uint32_t n = kOutW * kOutH;
    std::vector<uint8_t> rgb(n * 3);
    const float kVecScale = 8.0f;  // px → color
    for (uint32_t i = 0; i < n; ++i) {
        float fxpx = fv[i * 2 + 0] / 32.0f;  // S10.5 → pixels
        float fypx = fv[i * 2 + 1] / 32.0f;
        sumX += fxpx;
        sumY += fypx;
        auto clamp8 = [](float v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); };
        rgb[i * 3 + 0] = clamp8(128.0f + fxpx * kVecScale);
        rgb[i * 3 + 1] = clamp8(128.0f + fypx * kVecScale);
        rgb[i * 3 + 2] = 128;
    }
    float meanX = (float)(sumX / n), meanY = (float)(sumY / n);
    std::printf("ground truth motion: (%d, %d) px\n", kShiftX, kShiftY);
    std::printf("mean measured flow : (%.3f, %.3f) px  [grid %ux%u, %u vectors]\n",
                meanX, meanY, kOutW, kOutH, n);

    FILE* f = std::fopen(outPath, "wb");
    if (f) {
        std::fprintf(f, "P6\n%u %u\n255\n", kOutW, kOutH);
        std::fwrite(rgb.data(), 1, rgb.size(), f);
        std::fclose(f);
        std::printf("wrote %s (%ux%u, R=+x G=+y around 128)\n", outPath, kOutW, kOutH);
    }

    bool ok = std::fabs(meanX - kShiftX) < 2.0f && std::fabs(meanY - kShiftY) < 2.0f;
    std::printf("RESULT: %s\n", ok ? "PASS (measured flow matches ground truth)"
                                   : "CHECK (measured flow differs from ground truth)");

    // ---- teardown --------------------------------------------------------
    vkUnmapMemory(dev, rbMem);
    vkDestroyBuffer(dev, rb, nullptr);
    vkFreeMemory(dev, rbMem, nullptr);
    if (pDestroySession) pDestroySession(dev, session, nullptr);
    for (Image* im : {&input, &reference, &flow}) {
        vkDestroyImageView(dev, im->view, nullptr);
        vkDestroyImage(dev, im->image, nullptr);
        vkFreeMemory(dev, im->memory, nullptr);
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(instance, nullptr);
    return ok ? 0 : 3;
}
