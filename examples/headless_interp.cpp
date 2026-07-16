// M1 headless example + validation for libnvofg.
//
// Builds a synthetic frame pair with a known motion (curr = prev shifted +8px),
// runs nvofg 2x interpolation (phase 0.5) through the public C ABI, waits on the
// returned timeline point, reads the interpolated image back, and checks it
// against the analytic midpoint (pattern shifted +4px). Writes a PPM.
//
// At phase 0.5 the two-sided warp samples prev at p-4 and curr at p+4, both of
// which equal pattern(p-4) for a pure translation — so the interpolated frame
// should match the +4px ground truth in textured, non-border regions.

#include "nvofg.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define VKCHECK(e) do { VkResult _r=(e); if(_r){ std::fprintf(stderr,"%s -> %d\n",#e,(int)_r); return 2; } } while(0)

static const uint32_t W = 256, H = 256;
static const int SHIFT = 8;

static uint8_t pat(int x, int y) {
    float v = 0.5f + 0.25f * std::sin(x * 0.20f) + 0.25f * std::sin(y * 0.17f);
    if (((x >> 4) ^ (y >> 4)) & 1) v = 1.0f - v;
    int iv = (int)(v * 255.0f);
    return (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
}

static uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return UINT32_MAX;
}

struct Img { VkImage image; VkDeviceMemory mem; VkImageView view; };

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "interp.ppm";

    VkApplicationInfo app{}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    VkInstance instance; VKCHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t n = 0; vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(instance, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (auto c : pds) {
        uint32_t ec = 0; vkEnumerateDeviceExtensionProperties(c, nullptr, &ec, nullptr);
        std::vector<VkExtensionProperties> e(ec); vkEnumerateDeviceExtensionProperties(c, nullptr, &ec, e.data());
        for (auto& x : e) if (!std::strcmp(x.extensionName, VK_NV_OPTICAL_FLOW_EXTENSION_NAME)) pd = c;
    }
    if (!pd) { std::printf("SKIP: no OFA device\n"); return 0; }

    uint32_t gfxFamily = 0, ofFamily = 0;
    VKCHECK(nvofg_optical_flow_queue_family(instance, pd, vkGetInstanceProcAddr, &ofFamily) ? VK_ERROR_FEATURE_NOT_PRESENT : VK_SUCCESS);
    { uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,nullptr);
      std::vector<VkQueueFamilyProperties> q(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,q.data());
      for (uint32_t i=0;i<qn;++i) if (q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT){gfxFamily=i;break;} }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qc[2]{};
    qc[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; qc[0].queueFamilyIndex = gfxFamily; qc[0].queueCount = 1; qc[0].pQueuePriorities = &prio;
    qc[1] = qc[0]; qc[1].queueFamilyIndex = ofFamily;
    uint32_t qcCount = (gfxFamily == ofFamily) ? 1 : 2;

    VkPhysicalDeviceOpticalFlowFeaturesNV of{}; of.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV; of.opticalFlow = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{}; v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES; v12.timelineSemaphore = VK_TRUE; v12.pNext = &of;
    VkPhysicalDeviceFeatures2 f2{}; f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE; f2.pNext = &v12;

    uint32_t reqN = 0; const char* const* reqE = nvofg_required_device_extensions(&reqN);
    VkDeviceCreateInfo dci{}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.pNext = &f2;
    dci.queueCreateInfoCount = qcCount; dci.pQueueCreateInfos = qc;
    dci.enabledExtensionCount = reqN; dci.ppEnabledExtensionNames = reqE;
    VkDevice dev; VKCHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue gfxQ, ofQ; vkGetDeviceQueue(dev, gfxFamily, 0, &gfxQ); vkGetDeviceQueue(dev, ofFamily, 0, &ofQ);

    // --- create prev/curr color (RGBA8) + output (RGBA8) ---
    auto mkImg = [&](VkFormat fmt, VkImageUsageFlags usage) {
        Img im{};
        VkImageCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt; ci.extent = {W,H,1};
        ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateImage(dev, &ci, nullptr, &im.image);
        VkMemoryRequirements rq{}; vkGetImageMemoryRequirements(dev, im.image, &rq);
        VkMemoryAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize = rq.size;
        ai.memoryTypeIndex = memType(pd, rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(dev, &ai, nullptr, &im.mem); vkBindImageMemory(dev, im.image, im.mem, 0);
        VkImageViewCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; vi.image = im.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        vkCreateImageView(dev, &vi, nullptr, &im.view); return im;
    };
    Img prev = mkImg(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img curr = mkImg(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img outI = mkImg(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    Img uiImg = mkImg(VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img mvImg = mkImg(VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    // UI mask rectangle (never interpolated).
    const uint32_t UIX0=96, UIX1=160, UIY0=96, UIY1=160;

    // --- staging upload of the pair ---
    VkCommandPool pool; { VkCommandPoolCreateInfo p{}; p.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; p.queueFamilyIndex = gfxFamily; p.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; vkCreateCommandPool(dev,&p,nullptr,&pool); }
    auto oneShot = [&](auto rec){ VkCommandBufferAllocateInfo a{}; a.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; a.commandPool=pool; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer c; vkAllocateCommandBuffers(dev,&a,&c); VkCommandBufferBeginInfo b{}; b.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(c,&b); rec(c); vkEndCommandBuffer(c); VkSubmitInfo s{}; s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; s.commandBufferCount=1; s.pCommandBuffers=&c; vkQueueSubmit(gfxQ,1,&s,VK_NULL_HANDLE); vkQueueWaitIdle(gfxQ); vkFreeCommandBuffers(dev,pool,1,&c); };

    auto upload = [&](Img& im, int shift){
        std::vector<uint8_t> px(W*H*4);
        for (uint32_t y=0;y<H;++y) for (uint32_t x=0;x<W;++x){ uint8_t v=pat((int)x-shift,(int)y); uint8_t* d=&px[(y*W+x)*4]; d[0]=d[1]=d[2]=v; d[3]=255; }
        VkBuffer buf; VkDeviceMemory bm; VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=px.size(); bc.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&buf);
        VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,buf,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&bm); vkBindBufferMemory(dev,buf,bm,0);
        void* m; vkMapMemory(dev,bm,0,px.size(),0,&m); std::memcpy(m,px.data(),px.size()); vkUnmapMemory(dev,bm);
        oneShot([&](VkCommandBuffer c){
            VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=im.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={W,H,1};
            vkCmdCopyBufferToImage(c,buf,im.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&r);
            b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout=VK_IMAGE_LAYOUT_GENERAL; b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT;
            vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
        });
        vkDestroyBuffer(dev,buf,nullptr); vkFreeMemory(dev,bm,nullptr);
    };
    upload(prev, 0); upload(curr, SHIFT);

    // upload the UI mask (R8: 255 inside the rectangle, 0 elsewhere)
    {
        std::vector<uint8_t> m(W*H, 0);
        for (uint32_t y=UIY0;y<UIY1;++y) for (uint32_t x=UIX0;x<UIX1;++x) m[y*W+x]=255;
        VkBuffer buf; VkDeviceMemory bm; VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=m.size(); bc.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&buf);
        VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,buf,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&bm); vkBindBufferMemory(dev,buf,bm,0);
        void* mp; vkMapMemory(dev,bm,0,m.size(),0,&mp); std::memcpy(mp,m.data(),m.size()); vkUnmapMemory(dev,bm);
        oneShot([&](VkCommandBuffer c){
            VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=uiImg.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={W,H,1};
            vkCmdCopyBufferToImage(c,buf,uiImg.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&r);
            b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout=VK_IMAGE_LAYOUT_GENERAL; b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT;
            vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
        });
        vkDestroyBuffer(dev,buf,nullptr); vkFreeMemory(dev,bm,nullptr);
    }

    // upload synthetic motion vectors (RG32F): (+8,0) everywhere, = the true motion
    {
        std::vector<float> mv(W*H*2); for (uint32_t i=0;i<W*H;++i){ mv[i*2]=(float)SHIFT; mv[i*2+1]=0.f; }
        VkBuffer buf; VkDeviceMemory bm; VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=mv.size()*4; bc.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&buf);
        VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,buf,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&bm); vkBindBufferMemory(dev,buf,bm,0);
        void* mp; vkMapMemory(dev,bm,0,mv.size()*4,0,&mp); std::memcpy(mp,mv.data(),mv.size()*4); vkUnmapMemory(dev,bm);
        oneShot([&](VkCommandBuffer c){
            VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=mvImg.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={W,H,1};
            vkCmdCopyBufferToImage(c,buf,mvImg.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&r);
            b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout=VK_IMAGE_LAYOUT_GENERAL; b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT;
            vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
        });
        vkDestroyBuffer(dev,buf,nullptr); vkFreeMemory(dev,bm,nullptr);
    }

    // --- create nvofg + register + generate ---
    NvofgCreateInfo ci{}; ci.instance=instance; ci.physical_device=pd; ci.device=dev; ci.queue=gfxQ; ci.queue_family_index=gfxFamily;
    ci.of_queue=ofQ; ci.of_queue_family_index=ofFamily; ci.gipa=vkGetInstanceProcAddr; ci.width=W; ci.height=H;
    ci.quality=NVOFG_QUALITY_HIGH; ci.interpolator=NVOFG_INTERP_WARP; ci.mode=NVOFG_MODE_AUTOMATIC;
    ci.flags = NVOFG_FLAG_USE_UI_MASK | NVOFG_FLAG_BIDIRECTIONAL | NVOFG_FLAG_USE_MOTION;
    NvofgContext* ctx=nullptr;
    if (nvofg_create(&ci,&ctx)!=NVOFG_OK){ std::fprintf(stderr,"nvofg_create failed\n"); return 3; }

    NvofgImageDesc pd0{prev.image,prev.view,VK_FORMAT_R8G8B8A8_UNORM,W,H};
    NvofgImageDesc cd0{curr.image,curr.view,VK_FORMAT_R8G8B8A8_UNORM,W,H};
    NvofgImageDesc od0{outI.image,outI.view,VK_FORMAT_R8G8B8A8_UNORM,W,H};
    NvofgImageDesc ui0{uiImg.image,uiImg.view,VK_FORMAT_R8_UNORM,W,H};
    if (nvofg_register_color(ctx,&pd0,&cd0)!=NVOFG_OK || nvofg_register_output(ctx,&od0)!=NVOFG_OK){ std::fprintf(stderr,"register failed\n"); return 3; }
    NvofgImageDesc mv0{mvImg.image,mvImg.view,VK_FORMAT_R32G32_SFLOAT,W,H};
    NvofgAuxDesc aux{}; aux.ui_mask=&ui0; aux.motion=&mv0;
    if (nvofg_register_aux(ctx,&aux)!=NVOFG_OK){ std::fprintf(stderr,"register_aux failed\n"); return 3; }

    // Debug visualisation target: request the forward flow view.
    Img dbgImg = mkImg(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    NvofgImageDesc dbg0{dbgImg.image,dbgImg.view,VK_FORMAT_R8G8B8A8_UNORM,W,H};
    nvofg_set_debug_view(ctx, NVOFG_DEBUG_FLOW_FWD, &dbg0);

    NvofgGenerateInfo gi{}; gi.phase=0.5f; gi.prev_layout=VK_IMAGE_LAYOUT_GENERAL; gi.curr_layout=VK_IMAGE_LAYOUT_GENERAL;
    gi.input_timeline=VK_NULL_HANDLE;  // colors already uploaded+idle
    NvofgFrameSync sync{};
    // Several frames to exercise the command-buffer ring (slot reuse, pool reset).
    for (int frame=0; frame<5; ++frame) {
        NvofgResult gr = nvofg_record_generate(ctx,&gi,&sync);
        if (gr!=NVOFG_OK){ std::fprintf(stderr,"record_generate -> %d: %s\n",(int)gr,nvofg_last_error(ctx)); return 3; }
        VkSemaphoreWaitInfo wi{}; wi.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO; wi.semaphoreCount=1; wi.pSemaphores=&sync.semaphore; wi.pValues=&sync.value;
        VKCHECK(vkWaitSemaphores(dev,&wi,UINT64_MAX));
    }

    // Resize round-trip: tear down + rebuild the pipeline, re-register, regenerate.
    // (Same extent here for a compact test; a different extent takes the same path.)
    if (nvofg_resize(ctx, W, H) != NVOFG_OK) { std::fprintf(stderr,"resize failed: %s\n", nvofg_last_error(ctx)); return 3; }
    if (nvofg_register_color(ctx,&pd0,&cd0)!=NVOFG_OK || nvofg_register_output(ctx,&od0)!=NVOFG_OK
        || nvofg_register_aux(ctx,&aux)!=NVOFG_OK){ std::fprintf(stderr,"re-register failed\n"); return 3; }
    nvofg_set_debug_view(ctx, NVOFG_DEBUG_FLOW_FWD, &dbg0);
    if (nvofg_record_generate(ctx,&gi,&sync)!=NVOFG_OK){ std::fprintf(stderr,"post-resize generate failed: %s\n", nvofg_last_error(ctx)); return 3; }
    { VkSemaphoreWaitInfo wi{}; wi.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO; wi.semaphoreCount=1; wi.pSemaphores=&sync.semaphore; wi.pValues=&sync.value; VKCHECK(vkWaitSemaphores(dev,&wi,UINT64_MAX)); }
    std::printf("resize round-trip: OK\n");

    // --- read output back ---
    VkBuffer rb; VkDeviceMemory rbm; VkDeviceSize bytes=(VkDeviceSize)W*H*4;
    { VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=bytes; bc.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&rb);
      VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,rb,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&rbm); vkBindBufferMemory(dev,rb,rbm,0); }
    oneShot([&](VkCommandBuffer c){
        VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=VK_IMAGE_LAYOUT_GENERAL; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=outI.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT; b.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
        VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={W,H,1};
        vkCmdCopyImageToBuffer(c,outI.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,rb,1,&r);
    });
    uint8_t* px; vkMapMemory(dev,rbm,0,bytes,0,(void**)&px);

    // --- validate against the +4px analytic midpoint (interior only) ---
    double mae=0; int cnt=0; const int M=SHIFT+4;  // ignore border where flow falls off
    auto inUI=[&](uint32_t x,uint32_t y){ return x>=UIX0&&x<UIX1&&y>=UIY0&&y<UIY1; };
    for (uint32_t y=(uint32_t)M;y<H-M;++y) for (uint32_t x=(uint32_t)M;x<W-M;++x){
        if (inUI(x,y)) continue;                     // UI region is not interpolated
        int got = px[(y*W+x)*4]; int gt = pat((int)x-4,(int)y);
        mae += std::abs(got-gt); ++cnt;
    }
    mae /= (cnt?cnt:1);
    std::printf("interpolated vs +4px ground truth: MAE = %.2f (0-255) over %d interior px\n", mae, cnt);

    // UI mask: masked pixels must equal curr (= pattern shifted +8), NOT the +4 midpoint.
    double uiMaeCurr=0, uiMaeMid=0; int uiCnt=0;
    for (uint32_t y=UIY0+2;y<UIY1-2;++y) for (uint32_t x=UIX0+2;x<UIX1-2;++x){
        int got=px[(y*W+x)*4];
        uiMaeCurr += std::abs(got - pat((int)x-SHIFT,(int)y));
        uiMaeMid  += std::abs(got - pat((int)x-4,(int)y));
        ++uiCnt;
    }
    uiMaeCurr/=uiCnt; uiMaeMid/=uiCnt;
    std::printf("UI region: MAE vs curr = %.2f, MAE vs midpoint = %.2f (want curr<<midpoint)\n", uiMaeCurr, uiMaeMid);
    bool uiOk = uiMaeCurr < 3.0 && uiMaeMid > uiMaeCurr + 5.0;

    FILE* f=std::fopen(outPath,"wb");
    if (f){ std::fprintf(f,"P6\n%u %u\n255\n",W,H); std::vector<uint8_t> rgb(W*H*3); for(uint32_t i=0;i<W*H;++i){rgb[i*3]=px[i*4];rgb[i*3+1]=px[i*4+1];rgb[i*3+2]=px[i*4+2];} std::fwrite(rgb.data(),1,rgb.size(),f); std::fclose(f); std::printf("wrote %s\n",outPath); }

    bool ok = mae < 12.0 && uiOk;  // warp+bilinear+flow-noise tolerance + UI mask honoured
    std::printf("RESULT: %s\n", ok ? "PASS (midpoint matches + UI mask honoured)" : "CHECK (see MAE above)");

    vkUnmapMemory(dev,rbm); vkDestroyBuffer(dev,rb,nullptr); vkFreeMemory(dev,rbm,nullptr);
    nvofg_destroy(ctx);
    for (Img* im : {&prev,&curr,&outI,&uiImg,&mvImg,&dbgImg}){ vkDestroyImageView(dev,im->view,nullptr); vkDestroyImage(dev,im->image,nullptr); vkFreeMemory(dev,im->mem,nullptr); }
    vkDestroyCommandPool(dev,pool,nullptr); vkDestroyDevice(dev,nullptr); vkDestroyInstance(instance,nullptr);
    return ok ? 0 : 4;
}
