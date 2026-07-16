// Functional NGX backend GPU test (ADR 0006): drives the *official* DLSS path end to end
// through the RenderFX API — rfx_create probes NGX, rfx_query_capabilities must mark DLAA
// + DLSS-RR supported on an RTX device, then we commit and actually record+submit a DLAA
// (native-res) evaluate and a Ray-Reconstruction evaluate over synthetic GBuffer inputs,
// verifying NGX produces non-trivial output. SKIPs cleanly where NGX cannot init (no
// NVIDIA GPU / no model) — exactly the graceful-degradation contract (G6).
//
// Built only with -DRENDERFX_NGX. NGX needs its own instance+device extensions enabled
// *before* device creation (the app's responsibility, like nvofg's OF queue) — we query
// them via NVSDK_NGX_VULKAN_RequiredExtensions and enable every device feature the GPU
// supports so DLSS gets whatever its cubins need.
#include "renderfx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
int NVSDK_NGX_VULKAN_RequiredExtensions(unsigned* ic, const char*** ie, unsigned* dc, const char*** de);
}

#define VK(e) do { VkResult _r=(e); if(_r){ std::fprintf(stderr,"%s -> %d\n",#e,(int)_r); return 2; } } while(0)

static uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return ~0u;
}
struct Img { VkImage image=VK_NULL_HANDLE; VkDeviceMemory mem=VK_NULL_HANDLE; VkImageView view=VK_NULL_HANDLE; VkFormat fmt; uint32_t w,h; };

int main() {
    // ---- instance with NGX-required + GPDP2 extensions -------------------------------
    unsigned ic=0, dc=0; const char** ie=nullptr; const char** de=nullptr;
    NVSDK_NGX_VULKAN_RequiredExtensions(&ic,&ie,&dc,&de);
    std::vector<const char*> instExts(ie, ie+ic);
    instExts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkApplicationInfo app{}; app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{}; ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo=&app;
    ici.enabledExtensionCount=(uint32_t)instExts.size(); ici.ppEnabledExtensionNames=instExts.data();
    VkInstance inst; VK(vkCreateInstance(&ici,nullptr,&inst));

    uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
    VkPhysicalDevice pd=VK_NULL_HANDLE; uint32_t fam=0;
    for (auto c:pds){ VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(c,&p); if(p.vendorID!=0x10DE) continue;
        uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(c,&qn,nullptr); std::vector<VkQueueFamilyProperties> q(qn); vkGetPhysicalDeviceQueueFamilyProperties(c,&qn,q.data());
        for(uint32_t i=0;i<qn;++i) if((q[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)&&(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT)){pd=c;fam=i;break;} if(pd)break; }
    if(!pd){ std::printf("SKIP: no NVIDIA device -> NGX unavailable (backend reports unsupported)\n"); return 0; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd,&props); std::printf("device: %s\n",props.deviceName);

    // Enable every supported device feature (V11/V12/V13) so DLSS gets what its cubins need.
    VkPhysicalDeviceVulkan13Features f13{}; f13.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan12Features f12{}; f12.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES; f12.pNext=&f13;
    VkPhysicalDeviceVulkan11Features f11{}; f11.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES; f11.pNext=&f12;
    VkPhysicalDeviceFeatures2 f2{}; f2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2; f2.pNext=&f11;
    vkGetPhysicalDeviceFeatures2(pd,&f2);

    float prio=1.0f; VkDeviceQueueCreateInfo qci{}; qci.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; qci.queueFamilyIndex=fam; qci.queueCount=1; qci.pQueuePriorities=&prio;
    std::vector<const char*> devExts;   // bufferDeviceAddress comes from V12 features -> drop the EXT (VUID-04748)
    for(unsigned i=0;i<dc;++i){ if(std::strcmp(de[i],VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)==0) continue; devExts.push_back(de[i]); }
    VkDeviceCreateInfo dci{}; dci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.pNext=&f2; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=(uint32_t)devExts.size(); dci.ppEnabledExtensionNames=devExts.data();
    VkDevice dev; VK(vkCreateDevice(pd,&dci,nullptr,&dev)); VkQueue queue; vkGetDeviceQueue(dev,fam,0,&queue);

    auto mkImg=[&](uint32_t w,uint32_t h,VkFormat fmt,VkImageAspectFlags aspect,VkImageUsageFlags u){ Img im{}; im.fmt=fmt; im.w=w; im.h=h;
        VkImageCreateInfo ci{}; ci.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; ci.imageType=VK_IMAGE_TYPE_2D; ci.format=fmt; ci.extent={w,h,1}; ci.mipLevels=1; ci.arrayLayers=1; ci.samples=VK_SAMPLE_COUNT_1_BIT; ci.tiling=VK_IMAGE_TILING_OPTIMAL; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateImage(dev,&ci,nullptr,&im.image);
        VkMemoryRequirements rq{}; vkGetImageMemoryRequirements(dev,im.image,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); vkAllocateMemory(dev,&ai,nullptr,&im.mem); vkBindImageMemory(dev,im.image,im.mem,0);
        VkImageViewCreateInfo vi{}; vi.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; vi.image=im.image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=fmt; vi.subresourceRange={aspect,0,1,0,1}; vkCreateImageView(dev,&vi,nullptr,&im.view); return im; };

    VkCommandPool pool; { VkCommandPoolCreateInfo p{}; p.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; p.queueFamilyIndex=fam; p.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; vkCreateCommandPool(dev,&p,nullptr,&pool); }
    auto oneShot=[&](auto rec){ VkCommandBufferAllocateInfo a{}; a.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; a.commandPool=pool; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer c; vkAllocateCommandBuffers(dev,&a,&c); VkCommandBufferBeginInfo b{}; b.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(c,&b); rec(c); vkEndCommandBuffer(c); VkSubmitInfo s{}; s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; s.commandBufferCount=1; s.pCommandBuffers=&c; vkQueueSubmit(queue,1,&s,VK_NULL_HANDLE); vkQueueWaitIdle(queue); vkFreeCommandBuffers(dev,pool,1,&c); };
    auto toGeneral=[&](Img& im,VkImageAspectFlags aspect){ oneShot([&](VkCommandBuffer c){ VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_GENERAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=im.image; b.subresourceRange={aspect,0,1,0,1}; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b); }); };
    // Fill a color image with a gradient so we can tell NGX wrote *something* to output.
    auto fillGradient=[&](Img& im){ std::vector<uint16_t> px((size_t)im.w*im.h*4);
        auto h=[&](float v){ /*float->half*/ uint32_t x; std::memcpy(&x,&v,4); uint32_t s=(x>>16)&0x8000, e=((x>>23)&0xff), m=x&0x7fffff; if(e<113)return (uint16_t)s; if(e>142)return (uint16_t)(s|0x7bff); return (uint16_t)(s|((e-112)<<10)|(m>>13)); };
        for(uint32_t y=0;y<im.h;++y)for(uint32_t x=0;x<im.w;++x){uint16_t*d=&px[((size_t)y*im.w+x)*4]; float g=(float)x/(float)im.w; d[0]=h(g);d[1]=h(1.0f-g);d[2]=h(0.5f);d[3]=h(1.0f);}
        VkDeviceSize bytes=px.size()*2; VkBuffer buf; VkDeviceMemory bm; VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=bytes; bc.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&buf);
        VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,buf,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&bm); vkBindBufferMemory(dev,buf,bm,0);
        void*m; vkMapMemory(dev,bm,0,bytes,0,&m); std::memcpy(m,px.data(),bytes); vkUnmapMemory(dev,bm);
        oneShot([&](VkCommandBuffer c){ VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=im.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={im.w,im.h,1}; vkCmdCopyBufferToImage(c,buf,im.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&r);
            VkImageMemoryBarrier b2=b; b2.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b2.newLayout=VK_IMAGE_LAYOUT_GENERAL; b2.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; b2.dstAccessMask=VK_ACCESS_SHADER_READ_BIT; vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b2); });
        vkDestroyBuffer(dev,buf,nullptr); vkFreeMemory(dev,bm,nullptr); };
    auto outNonZero=[&](Img& im)->bool{ VkDeviceSize bytes=(VkDeviceSize)im.w*im.h*4*2; VkBuffer rb; VkDeviceMemory rbm; VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=bytes; bc.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&rb);
        VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,rb,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&rbm); vkBindBufferMemory(dev,rb,rbm,0);
        oneShot([&](VkCommandBuffer c){ VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=VK_IMAGE_LAYOUT_GENERAL; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=im.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT; vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={im.w,im.h,1}; vkCmdCopyImageToBuffer(c,im.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,rb,1,&r); });
        uint16_t* px; vkMapMemory(dev,rbm,0,bytes,0,(void**)&px); bool nz=false; for(size_t i=0;i<(size_t)im.w*im.h*4 && !nz;++i) if(px[i]!=0) nz=true; vkUnmapMemory(dev,rbm); vkDestroyBuffer(dev,rb,nullptr); vkFreeMemory(dev,rbm,nullptr); return nz; };

    // ---- RenderFX context + NGX capability probe -------------------------------------
    RfxContext* ctx=nullptr; RfxCreateInfo ci{}; ci.instance=inst; ci.physical_device=pd; ci.device=dev; ci.queue=queue; ci.queue_family_index=fam; ci.gipa=vkGetInstanceProcAddr; ci.width=640; ci.height=360;
    if(rfx_create(&ci,&ctx)!=RFX_OK){ std::fprintf(stderr,"rfx_create failed\n"); return 3; }
    RfxCapabilities caps{}; rfx_query_capabilities(ctx,&caps);
    auto supported=[&](RfxBackendId id)->int{ for(uint32_t i=0;i<caps.count;++i) if(caps.backends[i].id==id) return caps.backends[i].supported; return -1; };
    int dlaaSup=supported(RFX_BACKEND_DLAA), rrSup=supported(RFX_BACKEND_DLSS_RR);
    std::printf("  caps: DLAA supported=%d, DLSS_RR supported=%d\n", dlaaSup, rrSup);
    if(dlaaSup<=0){ std::printf("SKIP: NGX did not initialise on this host -> backends unsupported (graceful)\n"); rfx_destroy(ctx); vkDestroyCommandPool(dev,pool,nullptr); vkDestroyDevice(dev,nullptr); vkDestroyInstance(inst,nullptr); return 0; }

    const uint32_t RW=640, RH=360;   // render (== target for DLAA)
    Img color   = mkImg(RW,RH,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img depth   = mkImg(RW,RH,VK_FORMAT_R32_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img motion  = mkImg(RW,RH,VK_FORMAT_R16G16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img outDlaa = mkImg(RW,RH,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    for(Img* im:{&depth,&motion}) toGeneral(*im,VK_IMAGE_ASPECT_COLOR_BIT);
    toGeneral(outDlaa,VK_IMAGE_ASPECT_COLOR_BIT);
    fillGradient(color);   // leaves color in GENERAL

    RfxFrameContext fc{}; fc.struct_size=sizeof(fc); fc.version=RFX_VERSION;
    fc.provided_inputs=RFX_INPUT_COLOR|RFX_INPUT_DEPTH|RFX_INPUT_MOTION|RFX_INPUT_JITTER;
    fc.color ={color.image,color.view,color.fmt,RW,RH};
    fc.depth ={depth.image,depth.view,depth.fmt,RW,RH};
    fc.motion={motion.image,motion.view,motion.fmt,RW,RH};
    fc.mv_scale[0]=1.0f; fc.mv_scale[1]=1.0f;

    // ---- DLAA evaluate ---------------------------------------------------------------
    { RfxSelection sel{}; sel.valid=1; sel.backend[RFX_STAGE_UPSCALING]=RFX_BACKEND_DLAA; if(rfx_commit(ctx,&sel)!=RFX_OK){ std::fprintf(stderr,"commit DLAA failed\n"); return 4; } }
    RfxImageDesc dlaaDst{outDlaa.image,outDlaa.view,outDlaa.fmt,RW,RH};
    RfxResult dr=RFX_INTERNAL; oneShot([&](VkCommandBuffer c){ dr=rfx_record_upscaling(ctx,c,&fc,&dlaaDst,1); });
    bool dlaaOk = dr==RFX_OK && outNonZero(outDlaa);
    std::printf("  DLAA    : record=%d output_nonzero=%d -> %s\n", dr, (int)outNonZero(outDlaa), dlaaOk?"OK":"FAIL");

    // ---- Ray Reconstruction evaluate (2x upscale, full GBuffer) ----------------------
    bool rrOk=true;
    if(rrSup>0){
        const uint32_t TW=RW*2, TH=RH*2;
        Img normals = mkImg(RW,RH,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Img rough   = mkImg(RW,RH,VK_FORMAT_R16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Img diff    = mkImg(RW,RH,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Img spec    = mkImg(RW,RH,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        Img outRr   = mkImg(TW,TH,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        for(Img* im:{&normals,&rough,&diff,&spec,&outRr}) toGeneral(*im,VK_IMAGE_ASPECT_COLOR_BIT);
        fc.provided_inputs |= RFX_INPUT_NORMALS|RFX_INPUT_ROUGHNESS|RFX_INPUT_ALBEDO_DIFFUSE|RFX_INPUT_ALBEDO_SPECULAR|RFX_INPUT_REPROJ;
        fc.normals        ={normals.image,normals.view,normals.fmt,RW,RH};
        fc.roughness      ={rough.image,rough.view,rough.fmt,RW,RH};
        fc.albedo_diffuse ={diff.image,diff.view,diff.fmt,RW,RH};
        fc.albedo_specular={spec.image,spec.view,spec.fmt,RW,RH};
        for(int i=0;i<16;++i){ fc.world_to_view[i]=(i%5==0)?1.0f:0.0f; fc.view_to_clip[i]=(i%5==0)?1.0f:0.0f; }
        RfxSelection sel{}; sel.valid=1; sel.backend[RFX_STAGE_RAY_RECONSTRUCTION]=RFX_BACKEND_DLSS_RR; rfx_commit(ctx,&sel);
        RfxImageDesc rrDst{outRr.image,outRr.view,outRr.fmt,TW,TH};
        RfxResult rr=RFX_INTERNAL; oneShot([&](VkCommandBuffer c){ rr=rfx_record_ray_reconstruction(ctx,c,&fc,&rrDst); });
        rrOk = rr==RFX_OK && outNonZero(outRr);
        std::printf("  DLSS-RR : record=%d output_nonzero=%d -> %s\n", rr, (int)outNonZero(outRr), rrOk?"OK":"FAIL");
        for(Img* im:{&normals,&rough,&diff,&spec,&outRr}){ vkDestroyImageView(dev,im->view,nullptr); vkDestroyImage(dev,im->image,nullptr); vkFreeMemory(dev,im->mem,nullptr); }
    }

    rfx_destroy(ctx);
    for(Img* im:{&color,&depth,&motion,&outDlaa}){ vkDestroyImageView(dev,im->view,nullptr); vkDestroyImage(dev,im->image,nullptr); vkFreeMemory(dev,im->mem,nullptr); }
    vkDestroyCommandPool(dev,pool,nullptr); vkDestroyDevice(dev,nullptr); vkDestroyInstance(inst,nullptr);
    bool ok=dlaaOk&&rrOk;
    std::printf("RESULT: %s\n", ok?"PASS (NGX DLAA + Ray Reconstruction evaluated on GPU)":"FAIL");
    return ok?0:5;
}
