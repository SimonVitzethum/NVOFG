//! Raw FFI bindings to the nvofg C ABI (`include/nvofg.h`).
//!
//! Hand-written against the header, using `ash::vk` handle/enum types (which are
//! `#[repr(transparent)]` over the raw Vulkan handles, so they are ABI-compatible
//! with the C `Vk*` types). Prefer the safe `nvofg` crate; use this directly only
//! for FFI interop.
#![allow(non_camel_case_types)]

use ash::vk;
use std::os::raw::c_char;

pub const NVOFG_VERSION_MAJOR: u32 = 0;
pub const NVOFG_VERSION_MINOR: u32 = 1;
pub const NVOFG_VERSION_PATCH: u32 = 0;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvofgResult {
    Ok = 0,
    Unsupported = 1,
    InvalidArgument = 2,
    NotRegistered = 3,
    DeviceLost = 4,
    OutOfMemory = 5,
    Internal = 6,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvofgQuality {
    Perf = 0,
    Balanced = 1,
    High = 2,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvofgMode {
    Automatic = 0,
    ExternalCommands = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvofgInterpolator {
    Warp = 0,
    Cnn = 1,
    Transformer = 2,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvofgDebugView {
    None = 0,
    FlowFwd = 1,
    FlowBwd = 2,
    Confidence = 3,
    Occlusion = 4,
    Disocclusion = 5,
}

// Creation / feature flags (NvofgCreateInfo.flags).
pub const NVOFG_FLAG_USE_DEPTH: u32 = 1 << 0;
pub const NVOFG_FLAG_USE_MOTION: u32 = 1 << 1;
pub const NVOFG_FLAG_BIDIRECTIONAL: u32 = 1 << 2;
pub const NVOFG_FLAG_HDR: u32 = 1 << 3;
pub const NVOFG_FLAG_USE_UI_MASK: u32 = 1 << 4;
pub const NVOFG_FLAG_USE_REACTIVE: u32 = 1 << 5;
pub const NVOFG_FLAG_USE_MATERIAL_ID: u32 = 1 << 6;

/// Opaque context handle.
#[repr(C)]
pub struct NvofgContext {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvofgCaps {
    pub supported: u32,
    pub bidirectional: u32,
    pub cost_map: u32,
    pub global_flow: u32,
    pub min_grid_size: u32,
    pub max_grid_size: u32,
    pub max_width: u32,
    pub max_height: u32,
    pub hint_support: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvofgCreateInfo {
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub of_queue: vk::Queue,
    pub of_queue_family_index: u32,
    pub gipa: vk::PFN_vkGetInstanceProcAddr,
    pub width: u32,
    pub height: u32,
    pub quality: NvofgQuality,
    pub interpolator: NvofgInterpolator,
    pub mode: NvofgMode,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvofgImageDesc {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvofgAuxDesc {
    pub depth: *const NvofgImageDesc,
    pub motion: *const NvofgImageDesc,
    pub ui_mask: *const NvofgImageDesc,
    pub reactive: *const NvofgImageDesc,
    pub material_id: *const NvofgImageDesc,
}

impl Default for NvofgAuxDesc {
    fn default() -> Self {
        Self {
            depth: std::ptr::null(),
            motion: std::ptr::null(),
            ui_mask: std::ptr::null(),
            reactive: std::ptr::null(),
            material_id: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NvofgFrameSync {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvofgGenerateInfo {
    pub phase: f32,
    pub reproj: [f32; 16],
    pub near_plane: f32,
    pub far_plane: f32,
    pub reset: u32,
    pub input_timeline: vk::Semaphore,
    pub input_value: u64,
    pub prev_layout: vk::ImageLayout,
    pub curr_layout: vk::ImageLayout,
    pub cmd: vk::CommandBuffer,
}

extern "C" {
    pub fn nvofg_query_support(
        instance: vk::Instance,
        pd: vk::PhysicalDevice,
        gipa: vk::PFN_vkGetInstanceProcAddr,
        out: *mut NvofgCaps,
    ) -> NvofgResult;

    pub fn nvofg_optical_flow_queue_family(
        instance: vk::Instance,
        pd: vk::PhysicalDevice,
        gipa: vk::PFN_vkGetInstanceProcAddr,
        out_family: *mut u32,
    ) -> NvofgResult;

    pub fn nvofg_required_device_extensions(out_count: *mut u32) -> *const *const c_char;

    pub fn nvofg_create(info: *const NvofgCreateInfo, out: *mut *mut NvofgContext) -> NvofgResult;
    pub fn nvofg_destroy(ctx: *mut NvofgContext);

    pub fn nvofg_register_color(
        ctx: *mut NvofgContext,
        prev: *const NvofgImageDesc,
        curr: *const NvofgImageDesc,
    ) -> NvofgResult;
    pub fn nvofg_register_aux(ctx: *mut NvofgContext, aux: *const NvofgAuxDesc) -> NvofgResult;
    pub fn nvofg_register_output(
        ctx: *mut NvofgContext,
        interpolated: *const NvofgImageDesc,
    ) -> NvofgResult;
    pub fn nvofg_unregister_all(ctx: *mut NvofgContext);
    pub fn nvofg_resize(ctx: *mut NvofgContext, width: u32, height: u32) -> NvofgResult;

    pub fn nvofg_record_generate(
        ctx: *mut NvofgContext,
        info: *const NvofgGenerateInfo,
        out_sync: *mut NvofgFrameSync,
    ) -> NvofgResult;

    pub fn nvofg_record_warp(
        ctx: *mut NvofgContext,
        phase: f32,
        wait_sem: vk::Semaphore,
        wait_val: u64,
        out_sync: *mut NvofgFrameSync,
    ) -> NvofgResult;

    pub fn nvofg_caps(ctx: *mut NvofgContext, out: *mut NvofgCaps) -> NvofgResult;
    pub fn nvofg_last_error(ctx: *mut NvofgContext) -> *const c_char;

    pub fn nvofg_set_debug_view(
        ctx: *mut NvofgContext,
        view: NvofgDebugView,
        target: *const NvofgImageDesc,
    ) -> NvofgResult;
}
