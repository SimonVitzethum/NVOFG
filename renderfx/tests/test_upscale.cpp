// RenderFX upscaling backends GPU test (roadmap M1 Native, M3 Temporal). Upscales an
// 8x8 horizontal ramp to 32x32 through rfx_record_upscaling, dispatching on the committed
// backend, and verifies a smooth, monotonic result. Temporal runs several accumulation
// frames (static scene -> converges to the upscale). Vendor-neutral; validation-clean.
#include "renderfx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define VK(e) do { VkResult _r=(e); if(_r){ std::fprintf(stderr,"%s -> %d\n",#e,(int)_r); return 2; } } while(0)
static const uint32_t SW = 8, SH = 8, DW = 32, DH = 32;

static uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return ~0u;
}
struct Img { VkImage image; VkDeviceMemory mem; VkImageView view; };

int main() {
    VkApplicationInfo app{}; app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{}; ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo=&app;
    VkInstance inst; VK(vkCreateInstance(&ici,nullptr,&inst));
    uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
    VkPhysicalDevice pd=VK_NULL_HANDLE; uint32_t fam=0;
    for (auto c:pds){ VkPhysicalDeviceFeatures f{}; vkGetPhysicalDeviceFeatures(c,&f); if(!f.shaderStorageImageWriteWithoutFormat) continue;
        uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(c,&qn,nullptr); std::vector<VkQueueFamilyProperties> q(qn); vkGetPhysicalDeviceQueueFamilyProperties(c,&qn,q.data());
        for(uint32_t i=0;i<qn;++i) if(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT){pd=c;fam=i;break;} if(pd)break; }
    if(!pd){ std::printf("SKIP: no compute + storage-write-without-format device\n"); return 0; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd,&props); std::printf("device: %s\n",props.deviceName);

    float prio=1.0f; VkDeviceQueueCreateInfo qci{}; qci.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; qci.queueFamilyIndex=fam; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkPhysicalDeviceFeatures feats{}; feats.shaderStorageImageWriteWithoutFormat=VK_TRUE;
    VkDeviceCreateInfo dci{}; dci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.pEnabledFeatures=&feats;
    VkDevice dev; VK(vkCreateDevice(pd,&dci,nullptr,&dev)); VkQueue queue; vkGetDeviceQueue(dev,fam,0,&queue);

    auto mkImg=[&](uint32_t w,uint32_t h,VkImageUsageFlags u){ Img im{};
        VkImageCreateInfo ci{}; ci.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; ci.imageType=VK_IMAGE_TYPE_2D; ci.format=VK_FORMAT_R8G8B8A8_UNORM; ci.extent={w,h,1}; ci.mipLevels=1; ci.arrayLayers=1; ci.samples=VK_SAMPLE_COUNT_1_BIT; ci.tiling=VK_IMAGE_TILING_OPTIMAL; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateImage(dev,&ci,nullptr,&im.image);
        VkMemoryRequirements rq{}; vkGetImageMemoryRequirements(dev,im.image,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); vkAllocateMemory(dev,&ai,nullptr,&im.mem); vkBindImageMemory(dev,im.image,im.mem,0);
        VkImageViewCreateInfo vi{}; vi.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; vi.image=im.image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=VK_FORMAT_R8G8B8A8_UNORM; vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; vkCreateImageView(dev,&vi,nullptr,&im.view); return im; };
    Img src=mkImg(SW,SH,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img dst=mkImg(DW,DH,VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    VkCommandPool pool; { VkCommandPoolCreateInfo p{}; p.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; p.queueFamilyIndex=fam; p.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; vkCreateCommandPool(dev,&p,nullptr,&pool); }
    auto oneShot=[&](auto rec){ VkCommandBufferAllocateInfo a{}; a.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; a.commandPool=pool; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer c; vkAllocateCommandBuffers(dev,&a,&c); VkCommandBufferBeginInfo b{}; b.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(c,&b); rec(c); vkEndCommandBuffer(c); VkSubmitInfo s{}; s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; s.commandBufferCount=1; s.pCommandBuffers=&c; vkQueueSubmit(queue,1,&s,VK_NULL_HANDLE); vkQueueWaitIdle(queue); vkFreeCommandBuffers(dev,pool,1,&c); };
    auto barrier=[&](VkCommandBuffer c,VkImage im,VkImageLayout o,VkImageLayout nw,VkAccessFlags sa,VkAccessFlags da){ VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=o; b.newLayout=nw; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=im; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.srcAccessMask=sa; b.dstAccessMask=da; vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b); };

    // upload src ramp (R=G=B=x*36), transition src+dst to GENERAL.
    { std::vector<uint8_t> px(SW*SH*4); for(uint32_t y=0;y<SH;++y)for(uint32_t x=0;x<SW;++x){uint8_t v=(uint8_t)(x*36);uint8_t*d=&px[(y*SW+x)*4];d[0]=d[1]=d[2]=v;d[3]=255;}
      VkBuffer buf; VkDeviceMemory bm; VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=px.size(); bc.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&buf);
      VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,buf,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&bm); vkBindBufferMemory(dev,buf,bm,0);
      void*m; vkMapMemory(dev,bm,0,px.size(),0,&m); std::memcpy(m,px.data(),px.size()); vkUnmapMemory(dev,bm);
      oneShot([&](VkCommandBuffer c){ barrier(c,src.image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT); VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={SW,SH,1}; vkCmdCopyBufferToImage(c,buf,src.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&r); barrier(c,src.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT); barrier(c,dst.image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,0,VK_ACCESS_SHADER_WRITE_BIT); });
      vkDestroyBuffer(dev,buf,nullptr); vkFreeMemory(dev,bm,nullptr); }

    RfxContext* ctx=nullptr; RfxCreateInfo ci{}; ci.instance=inst; ci.physical_device=pd; ci.device=dev; ci.queue=queue; ci.queue_family_index=fam; ci.gipa=vkGetInstanceProcAddr; ci.width=DW; ci.height=DH;
    if(rfx_create(&ci,&ctx)!=RFX_OK){ std::fprintf(stderr,"rfx_create failed\n"); return 3; }
    RfxFrameContext fc{}; fc.struct_size=sizeof(fc); fc.version=RFX_VERSION; fc.provided_inputs=RFX_INPUT_COLOR;
    fc.color={src.image,src.view,VK_FORMAT_R8G8B8A8_UNORM,SW,SH};
    RfxImageDesc dd{dst.image,dst.view,VK_FORMAT_R8G8B8A8_UNORM,DW,DH};

    VkBuffer rb; VkDeviceMemory rbm; VkDeviceSize bytes=(VkDeviceSize)DW*DH*4;
    { VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=bytes; bc.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&rb);
      VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,rb,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&rbm); vkBindBufferMemory(dev,rb,rbm,0); }

    auto commit=[&](RfxBackendId b){ RfxSelection sel{}; sel.valid=1; sel.backend[RFX_STAGE_UPSCALING]=b; rfx_commit(ctx,&sel); };
    auto verify=[&](const char* label)->bool{ // read dst -> check middle row is a smooth monotonic ramp
        oneShot([&](VkCommandBuffer c){ barrier(c,dst.image,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT); VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={DW,DH,1}; vkCmdCopyImageToBuffer(c,dst.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,rb,1,&r); barrier(c,dst.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_WRITE_BIT); });
        uint8_t* px; vkMapMemory(dev,rbm,0,bytes,0,(void**)&px); uint32_t y=DH/2; bool mono=true,smooth=true; int prev=-1; for(uint32_t x=0;x<DW;++x){int v=px[(y*DW+x)*4]; if(v<prev-2)mono=false; if(prev>=0&&v-prev>28)smooth=false; prev=v;} int first=px[(y*DW+0)*4],last=px[(y*DW+DW-1)*4]; vkUnmapMemory(dev,rbm);
        bool ok=mono&&smooth&&first<56&&last>190; std::printf("  %-8s: first=%d last=%d mono=%d smooth=%d -> %s\n",label,first,last,mono,smooth,ok?"OK":"FAIL"); return ok; };

    // Native (bilinear): 1 frame.
    commit(RFX_BACKEND_NATIVE);
    oneShot([&](VkCommandBuffer c){ rfx_record_upscaling(ctx,c,&fc,&dd,1); });
    bool nativeOk = verify("native");

    // Temporal (TAAU): several accumulation frames on a static scene.
    commit(RFX_BACKEND_TEMPORAL);
    for (int f=0; f<4; ++f) oneShot([&](VkCommandBuffer c){ rfx_record_upscaling(ctx,c,&fc,&dd,f==0?1u:0u); });
    bool tempOk = verify("temporal");

    vkDestroyBuffer(dev,rb,nullptr); vkFreeMemory(dev,rbm,nullptr);
    rfx_destroy(ctx);
    for(Img* im:{&src,&dst}){ vkDestroyImageView(dev,im->view,nullptr); vkDestroyImage(dev,im->image,nullptr); vkFreeMemory(dev,im->mem,nullptr); }
    vkDestroyCommandPool(dev,pool,nullptr); vkDestroyDevice(dev,nullptr); vkDestroyInstance(inst,nullptr);
    bool ok = nativeOk && tempOk;
    std::printf("RESULT: %s\n", ok?"PASS (native + temporal upscaling verified)":"FAIL");
    return ok?0:4;
}
