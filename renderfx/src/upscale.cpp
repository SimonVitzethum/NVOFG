// RenderFX Native upscaling backend (roadmap M1): a vendor-neutral bilinear upscale,
// recorded into the app's command buffer (RenderFX owns no render graph). This makes the
// Upscaling stage functional and proves the framework is multi-stage, not FG-only.
#include "renderfx_internal.hpp"

#include "rfx_spv_upscale_native.spv.h"   // embedded SPIR-V (generated)

namespace {

bool ensureUpscale(RfxContext* ctx) {
    if (ctx->upReady) return true;
    VkDevice d = ctx->info.device;

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = rfx_spv_upscale_native_size;
    smci.pCode = reinterpret_cast<const uint32_t*>(rfx_spv_upscale_native);
    if (vkCreateShaderModule(d, &smci, nullptr, &ctx->upSm) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 3; dl.pBindings = b;
    if (vkCreateDescriptorSetLayout(d, &dl, nullptr, &ctx->upSetLayout) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1; pl.pSetLayouts = &ctx->upSetLayout;
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(d, &pl, nullptr, &ctx->upPipeLayout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = ctx->upSm; cp.stage.pName = "main";
    cp.layout = ctx->upPipeLayout;
    if (vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &ctx->upPipeline) != VK_SUCCESS) return false;

    VkSamplerCreateInfo smp{};
    smp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    smp.magFilter = smp.minFilter = VK_FILTER_LINEAR;
    smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(d, &smp, nullptr, &ctx->upSampler);

    VkDescriptorPoolSize sz[3] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 1; dp.poolSizeCount = 3; dp.pPoolSizes = sz;
    if (vkCreateDescriptorPool(d, &dp, nullptr, &ctx->upPool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = ctx->upPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &ctx->upSetLayout;
    if (vkAllocateDescriptorSets(d, &ai, &ctx->upSet) != VK_SUCCESS) return false;

    ctx->upReady = true;
    return true;
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
    if (ctx->upSampler) vkDestroySampler(d, ctx->upSampler, nullptr);
    if (ctx->upSm) vkDestroyShaderModule(d, ctx->upSm, nullptr);
    ctx->upReady = false;
}
}  // namespace renderfx

extern "C" RfxResult rfx_record_upscaling(RfxContext* ctx, VkCommandBuffer cmd,
                                          const RfxImageDesc* src, const RfxImageDesc* dst) {
    if (!ctx || !cmd || !src || !dst) return RFX_INVALID_ARGUMENT;
    if (!ensureUpscale(ctx)) return RFX_INTERNAL;

    const VkImageLayout G = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo srcI{VK_NULL_HANDLE, src->view, G};
    VkDescriptorImageInfo sampI{ctx->upSampler, VK_NULL_HANDLE, G};
    VkDescriptorImageInfo dstI{VK_NULL_HANDLE, dst->view, G};
    VkWriteDescriptorSet w[3]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = ctx->upSet; w[0].dstBinding = 0;
    w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &srcI;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[1].dstSet = ctx->upSet; w[1].dstBinding = 1;
    w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &sampI;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[2].dstSet = ctx->upSet; w[2].dstBinding = 2;
    w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &dstI;
    vkUpdateDescriptorSets(ctx->info.device, 3, w, 0, nullptr);

    // Make prior writes to src/dst visible to the compute pass (both in GENERAL).
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                       VK_ACCESS_MEMORY_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);

    struct Push { uint32_t dstW, dstH; } push{dst->width, dst->height};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->upPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->upPipeLayout, 0, 1, &ctx->upSet, 0, nullptr);
    vkCmdPushConstants(cmd, ctx->upPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (dst->width + 7) / 8, (dst->height + 7) / 8, 1);
    return RFX_OK;
}
