// RenderFX upscaling backends (roadmap M1 Native, M3 Temporal). Records the *committed*
// upscaler into the app's command buffer (RenderFX owns no render graph). Native is a
// vendor-neutral bilinear upscale; Temporal (TAAU) accumulates a reprojected history
// (RenderFX-owned ping-pong) with neighborhood clamping. Both run on any Vulkan GPU.
#include "renderfx_internal.hpp"

#include <initializer_list>

#include "rfx_spv_upscale_native.spv.h"
#include "rfx_spv_upscale_temporal.spv.h"

namespace {

uint32_t memType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return ~0u;
}

bool createOwned(RfxContext* ctx, renderfx::OwnedImage& im, uint32_t w, uint32_t h,
                 VkFormat fmt, VkImageUsageFlags usage) {
    im.w = w; im.h = h;
    VkImageCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt; ci.extent = {w, h, 1};
    ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(ctx->info.device, &ci, nullptr, &im.image) != VK_SUCCESS) return false;
    VkMemoryRequirements rq{}; vkGetImageMemoryRequirements(ctx->info.device, im.image, &rq);
    VkMemoryAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize = rq.size;
    ai.memoryTypeIndex = memType(ctx->info.physical_device, rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(ctx->info.device, &ai, nullptr, &im.mem) != VK_SUCCESS) return false;
    vkBindImageMemory(ctx->info.device, im.image, im.mem, 0);
    VkImageViewCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(ctx->info.device, &vi, nullptr, &im.view) == VK_SUCCESS;
}

VkShaderModule makeModule(VkDevice d, const void* spv, size_t size) {
    VkShaderModuleCreateInfo smci{}; smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = size; smci.pCode = reinterpret_cast<const uint32_t*>(spv);
    VkShaderModule m = VK_NULL_HANDLE; vkCreateShaderModule(d, &smci, nullptr, &m); return m;
}

void ensureSampler(RfxContext* ctx) {
    if (ctx->upSampler) return;
    VkSamplerCreateInfo smp{}; smp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    smp.magFilter = smp.minFilter = VK_FILTER_LINEAR;
    smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(ctx->info.device, &smp, nullptr, &ctx->upSampler);
}

void memBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

// ---- Native (bilinear) ----------------------------------------------------
bool ensureNative(RfxContext* ctx) {
    if (ctx->upReady) return true;
    VkDevice d = ctx->info.device;
    ctx->upSm = makeModule(d, rfx_spv_upscale_native, rfx_spv_upscale_native_size);
    if (!ctx->upSm) return false;
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dl{}; dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 3; dl.pBindings = b;
    vkCreateDescriptorSetLayout(d, &dl, nullptr, &ctx->upSetLayout);
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1; pl.pSetLayouts = &ctx->upSetLayout; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(d, &pl, nullptr, &ctx->upPipeLayout);
    VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = ctx->upSm; cp.stage.pName = "main"; cp.layout = ctx->upPipeLayout;
    if (vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &ctx->upPipeline) != VK_SUCCESS) return false;
    ensureSampler(ctx);
    VkDescriptorPoolSize sz[3] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dp.maxSets = 1; dp.poolSizeCount = 3; dp.pPoolSizes = sz;
    vkCreateDescriptorPool(d, &dp, nullptr, &ctx->upPool);
    VkDescriptorSetAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; ai.descriptorPool = ctx->upPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &ctx->upSetLayout;
    vkAllocateDescriptorSets(d, &ai, &ctx->upSet);
    ctx->upReady = true;
    return true;
}

RfxResult recordNative(RfxContext* ctx, VkCommandBuffer cmd, const RfxImageDesc* src, const RfxImageDesc* dst) {
    if (!ensureNative(ctx)) return RFX_INTERNAL;
    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo si{VK_NULL_HANDLE, src->view, G}, pi{ctx->upSampler, VK_NULL_HANDLE, G}, di{VK_NULL_HANDLE, dst->view, G};
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ctx->upSet; w[i].dstBinding = i; w[i].descriptorCount = 1; }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &si;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &pi;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &di;
    vkUpdateDescriptorSets(ctx->info.device, 3, w, 0, nullptr);
    memBarrier(cmd);
    struct Push { uint32_t w, h; } push{dst->width, dst->height};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->upPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->upPipeLayout, 0, 1, &ctx->upSet, 0, nullptr);
    vkCmdPushConstants(cmd, ctx->upPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (dst->width + 7) / 8, (dst->height + 7) / 8, 1);
    return RFX_OK;
}

// ---- Temporal (TAAU) ------------------------------------------------------
bool ensureTemporal(RfxContext* ctx, uint32_t w, uint32_t h) {
    if (ctx->tReady) return true;
    VkDevice d = ctx->info.device;
    ctx->tSm = makeModule(d, rfx_spv_upscale_temporal, rfx_spv_upscale_temporal_size);
    if (!ctx->tSm) return false;
    VkDescriptorSetLayoutBinding b[6] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // current
        {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // history in
        {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // motion
        {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // dst
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};  // history out
    VkDescriptorSetLayoutCreateInfo dl{}; dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 6; dl.pBindings = b;
    vkCreateDescriptorSetLayout(d, &dl, nullptr, &ctx->tSetLayout);
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1; pl.pSetLayouts = &ctx->tSetLayout; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(d, &pl, nullptr, &ctx->tPipeLayout);
    VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = ctx->tSm; cp.stage.pName = "main"; cp.layout = ctx->tPipeLayout;
    if (vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &ctx->tPipeline) != VK_SUCCESS) return false;
    ensureSampler(ctx);
    if (!createOwned(ctx, ctx->history[0], w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) ||
        !createOwned(ctx, ctx->history[1], w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) ||
        !createOwned(ctx, ctx->dummyMotion, 1, 1, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;

    VkDescriptorPoolSize sz[3] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dp.maxSets = 1; dp.poolSizeCount = 3; dp.pPoolSizes = sz;
    vkCreateDescriptorPool(d, &dp, nullptr, &ctx->tPool);
    VkDescriptorSetAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; ai.descriptorPool = ctx->tPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &ctx->tSetLayout;
    vkAllocateDescriptorSets(d, &ai, &ctx->tSet);

    // one-shot: history/dummy -> GENERAL, dummy motion cleared to 0.
    VkCommandPool pool; VkCommandPoolCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; pci.queueFamilyIndex = ctx->info.queue_family_index;
    vkCreateCommandPool(d, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbi{}; cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer c; vkAllocateCommandBuffers(d, &cbi, &c);
    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(c, &bi);
    auto toGeneral = [&](VkImage im, VkImageLayout from, VkAccessFlags da) { VkImageMemoryBarrier b2{}; b2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; b2.oldLayout = from; b2.newLayout = VK_IMAGE_LAYOUT_GENERAL; b2.srcQueueFamilyIndex = b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b2.image = im; b2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; b2.dstAccessMask = da; vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b2); };
    toGeneral(ctx->history[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_SHADER_READ_BIT);
    toGeneral(ctx->history[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_SHADER_READ_BIT);
    toGeneral(ctx->dummyMotion.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkClearColorValue zero{}; VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(c, ctx->dummyMotion.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rng);
    vkEndCommandBuffer(c);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &c;
    vkQueueSubmit(ctx->info.queue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx->info.queue);
    vkDestroyCommandPool(d, pool, nullptr);

    ctx->tReady = true;
    return true;
}

RfxResult recordTemporal(RfxContext* ctx, VkCommandBuffer cmd, const RfxFrameContext* fc,
                         const RfxImageDesc* dst, uint32_t reset) {
    if (!ensureTemporal(ctx, dst->width, dst->height)) return RFX_INTERNAL;
    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL;
    const bool hasMotion = (fc->provided_inputs & RFX_INPUT_MOTION) != 0;
    const uint32_t inSlot = ctx->frameParity, outSlot = ctx->frameParity ^ 1u;

    VkDescriptorImageInfo cur{VK_NULL_HANDLE, fc->color.view, G};
    VkDescriptorImageInfo hin{VK_NULL_HANDLE, ctx->history[inSlot].view, G};
    VkDescriptorImageInfo mot{VK_NULL_HANDLE, hasMotion ? fc->motion.view : ctx->dummyMotion.view, G};
    VkDescriptorImageInfo samp{ctx->upSampler, VK_NULL_HANDLE, G};
    VkDescriptorImageInfo dsti{VK_NULL_HANDLE, dst->view, G};
    VkDescriptorImageInfo hout{VK_NULL_HANDLE, ctx->history[outSlot].view, G};
    VkDescriptorImageInfo* infos[6] = {&cur, &hin, &mot, &samp, &dsti, &hout};
    VkDescriptorType types[6] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                 VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER,
                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkWriteDescriptorSet w[6]{};
    for (int i = 0; i < 6; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ctx->tSet; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = types[i]; w[i].pImageInfo = infos[i]; }
    vkUpdateDescriptorSets(ctx->info.device, 6, w, 0, nullptr);

    memBarrier(cmd);
    uint32_t flags = 0;
    if (reset || !ctx->historyValid) flags |= 1u;
    if (hasMotion) flags |= 2u;
    struct Push { uint32_t w, h; float alpha; uint32_t flags; } push{dst->width, dst->height, 0.1f, flags};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->tPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->tPipeLayout, 0, 1, &ctx->tSet, 0, nullptr);
    vkCmdPushConstants(cmd, ctx->tPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (dst->width + 7) / 8, (dst->height + 7) / 8, 1);

    ctx->frameParity = outSlot;   // this frame's output becomes next frame's history
    ctx->historyValid = true;
    return RFX_OK;
}

}  // namespace

namespace renderfx {
void destroyUpscale(RfxContext* ctx) {
    VkDevice d = ctx->info.device;
    if (!d) return;
    if (ctx->upPipeline) vkDestroyPipeline(d, ctx->upPipeline, nullptr);
    if (ctx->upPipeLayout) vkDestroyPipelineLayout(d, ctx->upPipeLayout, nullptr);
    if (ctx->upSetLayout) vkDestroyDescriptorSetLayout(d, ctx->upSetLayout, nullptr);
    if (ctx->upPool) vkDestroyDescriptorPool(d, ctx->upPool, nullptr);
    if (ctx->upSm) vkDestroyShaderModule(d, ctx->upSm, nullptr);
    if (ctx->tPipeline) vkDestroyPipeline(d, ctx->tPipeline, nullptr);
    if (ctx->tPipeLayout) vkDestroyPipelineLayout(d, ctx->tPipeLayout, nullptr);
    if (ctx->tSetLayout) vkDestroyDescriptorSetLayout(d, ctx->tSetLayout, nullptr);
    if (ctx->tPool) vkDestroyDescriptorPool(d, ctx->tPool, nullptr);
    if (ctx->tSm) vkDestroyShaderModule(d, ctx->tSm, nullptr);
    for (OwnedImage* im : {&ctx->history[0], &ctx->history[1], &ctx->dummyMotion}) {
        if (im->view) vkDestroyImageView(d, im->view, nullptr);
        if (im->image) vkDestroyImage(d, im->image, nullptr);
        if (im->mem) vkFreeMemory(d, im->mem, nullptr);
    }
    if (ctx->upSampler) vkDestroySampler(d, ctx->upSampler, nullptr);
    ctx->upReady = ctx->tReady = false;
}
}  // namespace renderfx

extern "C" RfxResult rfx_record_upscaling(RfxContext* ctx, VkCommandBuffer cmd,
                                          const RfxFrameContext* fc, const RfxImageDesc* dst,
                                          uint32_t reset) {
    if (!ctx || !cmd || !fc || !dst) return RFX_INVALID_ARGUMENT;
    switch (ctx->upscaleBackend) {
        case RFX_BACKEND_TEMPORAL: return recordTemporal(ctx, cmd, fc, dst, reset);
        case RFX_BACKEND_DLAA:     return renderfx::ngxRecordDLAA(ctx, cmd, fc, dst, reset);
        case RFX_BACKEND_DLSS_SR:  return renderfx::ngxRecordDLAA(ctx, cmd, fc, dst, reset);
        case RFX_BACKEND_NONE:
        case RFX_BACKEND_NATIVE:   return recordNative(ctx, cmd, &fc->color, dst);
        default:                   return RFX_UNSUPPORTED;  // FSR/XeSS reserved
    }
}
