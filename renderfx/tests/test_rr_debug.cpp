// RenderFX RR/GBuffer debug-visualisation GPU test. Renders the normals channel of the
// Frame Context (verifies remap) and requests an absent channel (verifies the magenta
// missing-input diagnostic). Vendor-neutral; validation-clean.
#include "renderfx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define VK(e) do { VkResult _r=(e); if(_r){ std::fprintf(stderr,"%s -> %d\n",#e,(int)_r); return 2; } } while(0)
static const uint32_t W = 16, H = 16;

static uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i=0;i<mp.memoryTypeCount;++i) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p) return i;
    return ~0u;
}
struct Img { VkImage image; VkDeviceMemory mem; VkImageView view; };

int main() {
    VkApplicationInfo app{}; app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{}; ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo=&app;
    VkInstance inst; VK(vkCreateInstance(&ici,nullptr,&inst));
    uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
    VkPhysicalDevice pd=VK_NULL_HANDLE; uint32_t fam=0;
    for(auto c:pds){ VkPhysicalDeviceFeatures f{}; vkGetPhysicalDeviceFeatures(c,&f); if(!f.shaderStorageImageWriteWithoutFormat) continue; uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(c,&qn,nullptr); std::vector<VkQueueFamilyProperties> q(qn); vkGetPhysicalDeviceQueueFamilyProperties(c,&qn,q.data()); for(uint32_t i=0;i<qn;++i) if(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT){pd=c;fam=i;break;} if(pd)break; }
    if(!pd){ std::printf("SKIP: no suitable device\n"); return 0; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd,&props); std::printf("device: %s\n",props.deviceName);

    float prio=1.0f; VkDeviceQueueCreateInfo qci{}; qci.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; qci.queueFamilyIndex=fam; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkPhysicalDeviceFeatures feats{}; feats.shaderStorageImageWriteWithoutFormat=VK_TRUE;
    VkDeviceCreateInfo dci{}; dci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.pEnabledFeatures=&feats;
    VkDevice dev; VK(vkCreateDevice(pd,&dci,nullptr,&dev)); VkQueue queue; vkGetDeviceQueue(dev,fam,0,&queue);

    auto mkImg=[&](uint32_t w,uint32_t h,VkFormat fmt,VkImageUsageFlags u){ Img im{}; VkImageCreateInfo ci{}; ci.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; ci.imageType=VK_IMAGE_TYPE_2D; ci.format=fmt; ci.extent={w,h,1}; ci.mipLevels=1; ci.arrayLayers=1; ci.samples=VK_SAMPLE_COUNT_1_BIT; ci.tiling=VK_IMAGE_TILING_OPTIMAL; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateImage(dev,&ci,nullptr,&im.image); VkMemoryRequirements rq{}; vkGetImageMemoryRequirements(dev,im.image,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); vkAllocateMemory(dev,&ai,nullptr,&im.mem); vkBindImageMemory(dev,im.image,im.mem,0); VkImageViewCreateInfo vi{}; vi.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; vi.image=im.image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=fmt; vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; vkCreateImageView(dev,&vi,nullptr,&im.view); return im; };
    Img normals=mkImg(W,H,VK_FORMAT_R32G32B32A32_SFLOAT,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Img out=mkImg(W,H,VK_FORMAT_R8G8B8A8_UNORM,VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    VkCommandPool pool; { VkCommandPoolCreateInfo p{}; p.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; p.queueFamilyIndex=fam; p.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; vkCreateCommandPool(dev,&p,nullptr,&pool); }
    auto oneShot=[&](auto rec){ VkCommandBufferAllocateInfo a{}; a.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; a.commandPool=pool; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer c; vkAllocateCommandBuffers(dev,&a,&c); VkCommandBufferBeginInfo b{}; b.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(c,&b); rec(c); vkEndCommandBuffer(c); VkSubmitInfo s{}; s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; s.commandBufferCount=1; s.pCommandBuffers=&c; vkQueueSubmit(queue,1,&s,VK_NULL_HANDLE); vkQueueWaitIdle(queue); vkFreeCommandBuffers(dev,pool,1,&c); };
    auto barrier=[&](VkCommandBuffer c,VkImage im,VkImageLayout o,VkImageLayout nw,VkAccessFlags sa,VkAccessFlags da){ VkImageMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b.oldLayout=o; b.newLayout=nw; b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=im; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; b.srcAccessMask=sa; b.dstAccessMask=da; vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b); };

    // upload normals = (0,0,1) everywhere; transition normals + out to GENERAL.
    { std::vector<float> px(W*H*4); for(uint32_t i=0;i<W*H;++i){px[i*4+0]=0;px[i*4+1]=0;px[i*4+2]=1;px[i*4+3]=0;}
      VkBuffer buf; VkDeviceMemory bm; VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=px.size()*4; bc.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&buf);
      VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,buf,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&bm); vkBindBufferMemory(dev,buf,bm,0);
      void*m; vkMapMemory(dev,bm,0,px.size()*4,0,&m); std::memcpy(m,px.data(),px.size()*4); vkUnmapMemory(dev,bm);
      oneShot([&](VkCommandBuffer c){ barrier(c,normals.image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT); VkBufferImageCopy r{}; r.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; r.imageExtent={W,H,1}; vkCmdCopyBufferToImage(c,buf,normals.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&r); barrier(c,normals.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT); barrier(c,out.image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,0,VK_ACCESS_SHADER_WRITE_BIT); });
      vkDestroyBuffer(dev,buf,nullptr); vkFreeMemory(dev,bm,nullptr); }

    RfxContext* ctx=nullptr; RfxCreateInfo ci{}; ci.instance=inst; ci.physical_device=pd; ci.device=dev; ci.queue=queue; ci.queue_family_index=fam; ci.gipa=vkGetInstanceProcAddr; ci.width=W; ci.height=H;
    if(rfx_create(&ci,&ctx)!=RFX_OK){ std::fprintf(stderr,"rfx_create failed\n"); return 3; }
    RfxImageDesc od{out.image,out.view,VK_FORMAT_R8G8B8A8_UNORM,W,H};

    VkBuffer rb; VkDeviceMemory rbm; VkDeviceSize bytes=(VkDeviceSize)W*H*4;
    { VkBufferCreateInfo bc{}; bc.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size=bytes; bc.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT; bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(dev,&bc,nullptr,&rb); VkMemoryRequirements rq{}; vkGetBufferMemoryRequirements(dev,rb,&rq); VkMemoryAllocateInfo ai{}; ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize=rq.size; ai.memoryTypeIndex=memType(pd,rq.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); vkAllocateMemory(dev,&ai,nullptr,&rbm); vkBindBufferMemory(dev,rb,rbm,0); }
    auto center=[&](int* r,int* g,int* b){ oneShot([&](VkCommandBuffer c){ barrier(c,out.image,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT); VkBufferImageCopy rr{}; rr.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; rr.imageExtent={W,H,1}; vkCmdCopyImageToBuffer(c,out.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,rb,1,&rr); barrier(c,out.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_WRITE_BIT); }); uint8_t* px; vkMapMemory(dev,rbm,0,bytes,0,(void**)&px); uint32_t i=(H/2*W+W/2)*4; *r=px[i];*g=px[i+1];*b=px[i+2]; vkUnmapMemory(dev,rbm); };

    // Normals provided -> remapped (0,0,1) -> ~(128,128,255).
    RfxFrameContext fc{}; fc.struct_size=sizeof(fc); fc.version=RFX_VERSION; fc.provided_inputs=RFX_INPUT_NORMALS; fc.normals={normals.image,normals.view,VK_FORMAT_R32G32B32A32_SFLOAT,W,H};
    oneShot([&](VkCommandBuffer c){ rfx_record_debug_view(ctx,c,&fc,&od,RFX_DEBUG_NORMALS); });
    int r,g,b; center(&r,&g,&b); bool normalsOk = r>110&&r<145 && g>110&&g<145 && b>235;
    std::printf("  normals viz    : (%d,%d,%d) -> %s\n", r,g,b, normalsOk?"OK":"FAIL");

    // Roughness NOT provided -> magenta diagnostic (255,0,255).
    oneShot([&](VkCommandBuffer c){ rfx_record_debug_view(ctx,c,&fc,&od,RFX_DEBUG_ROUGHNESS); });
    center(&r,&g,&b); bool missingOk = r>235 && g<20 && b>235;
    std::printf("  missing input  : (%d,%d,%d) -> %s (magenta)\n", r,g,b, missingOk?"OK":"FAIL");

    vkDestroyBuffer(dev,rb,nullptr); vkFreeMemory(dev,rbm,nullptr);
    rfx_destroy(ctx);
    for(Img* im:{&normals,&out}){ vkDestroyImageView(dev,im->view,nullptr); vkDestroyImage(dev,im->image,nullptr); vkFreeMemory(dev,im->mem,nullptr); }
    vkDestroyCommandPool(dev,pool,nullptr); vkDestroyDevice(dev,nullptr); vkDestroyInstance(inst,nullptr);
    bool ok=normalsOk&&missingOk; std::printf("RESULT: %s\n", ok?"PASS (GBuffer viz + missing-input diagnostic)":"FAIL");
    return ok?0:4;
}
