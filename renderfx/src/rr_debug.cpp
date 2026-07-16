// RenderFX RR/GBuffer debug visualisation (vendor-neutral). Renders a Frame Context
// channel (normals/roughness/albedo/depth/motion) into an output image; magenta where the
// requested channel is absent (missing-input diagnostic). Records into the app's cmd buffer.
#include "renderfx_internal.hpp"

#include "rfx_spv_rr_debug.spv.h"

namespace {

uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return ~0u;
}

bool ensureDebug(RfxContext* ctx) {
    if (ctx->dbgReady) return true;
    VkDevice d = ctx->info.device;

    VkShaderModuleCreateInfo smci{}; smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = rfx_spv_rr_debug_size; smci.pCode = reinterpret_cast<const uint32_t*>(rfx_spv_rr_debug);
    if (vkCreateShaderModule(d, &smci, nullptr, &ctx->dbgSm) != VK_SUCCESS) return false;
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dl{}; dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dl.bindingCount = 3; dl.pBindings = b;
    vkCreateDescriptorSetLayout(d, &dl, nullptr, &ctx->dbgSetLayout);
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; pl.setLayoutCount = 1; pl.pSetLayouts = &ctx->dbgSetLayout; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(d, &pl, nullptr, &ctx->dbgPipeLayout);
    VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = ctx->dbgSm; cp.stage.pName = "main"; cp.layout = ctx->dbgPipeLayout;
    if (vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &ctx->dbgPipeline) != VK_SUCCESS) return false;
    if (!ctx->upSampler) { VkSamplerCreateInfo smp{}; smp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO; smp.magFilter = smp.minFilter = VK_FILTER_LINEAR; smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; vkCreateSampler(d, &smp, nullptr, &ctx->upSampler); }
    VkDescriptorPoolSize sz[3] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dp.maxSets = 1; dp.poolSizeCount = 3; dp.pPoolSizes = sz;
    vkCreateDescriptorPool(d, &dp, nullptr, &ctx->dbgPool);
    VkDescriptorSetAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; ai.descriptorPool = ctx->dbgPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &ctx->dbgSetLayout;
    vkAllocateDescriptorSets(d, &ai, &ctx->dbgSet);

    // 1x1 dummy source (bound when the requested channel is absent), -> GENERAL.
    renderfx::OwnedImage& im = ctx->dbgDummy; im.w = im.h = 1;
    VkImageCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; ci.imageType = VK_IMAGE_TYPE_2D; ci.format = VK_FORMAT_R8G8B8A8_UNORM; ci.extent = {1, 1, 1}; ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateImage(d, &ci, nullptr, &im.image);
    VkMemoryRequirements rq{}; vkGetImageMemoryRequirements(d, im.image, &rq); VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = rq.size; mai.memoryTypeIndex = memType(ctx->info.physical_device, rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); vkAllocateMemory(d, &mai, nullptr, &im.mem); vkBindImageMemory(d, im.image, im.mem, 0);
    VkImageViewCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; vi.image = im.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_UNORM; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; vkCreateImageView(d, &vi, nullptr, &im.view);
    VkCommandPool pool; VkCommandPoolCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; pci.queueFamilyIndex = ctx->info.queue_family_index; vkCreateCommandPool(d, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbi{}; cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1; VkCommandBuffer c; vkAllocateCommandBuffers(d, &cbi, &c);
    VkCommandBufferBeginInfo bg{}; bg.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; bg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(c, &bg);
    VkImageMemoryBarrier bar{}; bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; bar.newLayout = VK_IMAGE_LAYOUT_GENERAL; bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; bar.image = im.image; bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
    vkEndCommandBuffer(c); VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &c; vkQueueSubmit(ctx->info.queue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx->info.queue); vkDestroyCommandPool(d, pool, nullptr);

    ctx->dbgReady = true;
    return true;
}

// view -> (Frame Context channel, provided bit)
const RfxImageDesc* channelFor(const RfxFrameContext* fc, RfxDebugView v, uint32_t* bit) {
    switch (v) {
        case RFX_DEBUG_NORMALS:         *bit = RFX_INPUT_NORMALS;          return &fc->normals;
        case RFX_DEBUG_ROUGHNESS:       *bit = RFX_INPUT_ROUGHNESS;        return &fc->roughness;
        case RFX_DEBUG_DIFFUSE_ALBEDO:  *bit = RFX_INPUT_ALBEDO_DIFFUSE;   return &fc->albedo_diffuse;
        case RFX_DEBUG_SPECULAR_ALBEDO: *bit = RFX_INPUT_ALBEDO_SPECULAR;  return &fc->albedo_specular;
        case RFX_DEBUG_DEPTH:           *bit = RFX_INPUT_DEPTH;            return &fc->depth;
        case RFX_DEBUG_MOTION:          *bit = RFX_INPUT_MOTION;           return &fc->motion;
        default:                        *bit = 0;                          return nullptr;
    }
}

}  // namespace

namespace renderfx {
void destroyDebug(RfxContext* ctx) {
    VkDevice d = ctx->info.device;
    if (!d) return;
    if (ctx->dbgPipeline) vkDestroyPipeline(d, ctx->dbgPipeline, nullptr);
    if (ctx->dbgPipeLayout) vkDestroyPipelineLayout(d, ctx->dbgPipeLayout, nullptr);
    if (ctx->dbgSetLayout) vkDestroyDescriptorSetLayout(d, ctx->dbgSetLayout, nullptr);
    if (ctx->dbgPool) vkDestroyDescriptorPool(d, ctx->dbgPool, nullptr);
    if (ctx->dbgSm) vkDestroyShaderModule(d, ctx->dbgSm, nullptr);
    if (ctx->dbgDummy.view) vkDestroyImageView(d, ctx->dbgDummy.view, nullptr);
    if (ctx->dbgDummy.image) vkDestroyImage(d, ctx->dbgDummy.image, nullptr);
    if (ctx->dbgDummy.mem) vkFreeMemory(d, ctx->dbgDummy.mem, nullptr);
    ctx->dbgReady = false;
}
}  // namespace renderfx

extern "C" RfxResult rfx_record_debug_view(RfxContext* ctx, VkCommandBuffer cmd,
                                           const RfxFrameContext* fc, const RfxImageDesc* output,
                                           RfxDebugView view) {
    if (!ctx || !cmd || !fc || !output || view == RFX_DEBUG_NONE) return RFX_INVALID_ARGUMENT;
    if (!ensureDebug(ctx)) return RFX_INTERNAL;

    uint32_t bit = 0;
    const RfxImageDesc* chan = channelFor(fc, view, &bit);
    if (!chan) return RFX_INVALID_ARGUMENT;
    const uint32_t present = (fc->provided_inputs & bit) ? 1u : 0u;
    VkImageView srcView = present ? chan->view : ctx->dbgDummy.view;

    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo si{VK_NULL_HANDLE, srcView, G}, pi{ctx->upSampler, VK_NULL_HANDLE, G}, di{VK_NULL_HANDLE, output->view, G};
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ctx->dbgSet; w[i].dstBinding = i; w[i].descriptorCount = 1; }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &si;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &pi;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &di;
    vkUpdateDescriptorSets(ctx->info.device, 3, w, 0, nullptr);

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    struct Push { uint32_t w, h, mode, present; } push{output->width, output->height, (uint32_t)view, present};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->dbgPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->dbgPipeLayout, 0, 1, &ctx->dbgSet, 0, nullptr);
    vkCmdPushConstants(cmd, ctx->dbgPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (output->width + 7) / 8, (output->height + 7) / 8, 1);
    return RFX_OK;
}
