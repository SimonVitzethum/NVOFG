//! Safe Rust wrapper for **nvofg** — native Linux Vulkan frame generation on the
//! NVIDIA Optical Flow Accelerator (`VK_NV_optical_flow`).
//!
//! Accepts raw `ash::vk` handles so it works with any Vulkan backend (ash,
//! wgpu-hal, vulkano, …). [`FrameGen::new`] returns `Ok(None)` when the GPU/driver
//! has no usable OFA, so the 1× fallback is baked into the type system (goal G6).
//!
//! ```no_run
//! # use ash::vk; use nvofg::*;
//! # fn demo(ci: CreateInfo) -> Result<(), Error> {
//! let Some(mut fg) = FrameGen::new(&ci)? else { return Ok(()) /* run at 1x */ };
//! # let (prev, curr, out): (ImageDesc, ImageDesc, ImageDesc) = todo!();
//! fg.register_color(&prev, &curr)?;
//! fg.register_output(&out)?;
//! let sync = fg.record_generate(&GenerateInfo { phase: 0.5, ..Default::default() })?;
//! // wait on (sync.semaphore, sync.value) before presenting the interpolated frame
//! # Ok(()) }
//! ```
#![deny(unsafe_op_in_unsafe_fn)]

use ash::vk;
use nvofg_sys as sys;
use std::ffi::CStr;
use std::fmt;

pub use sys::{
    NvofgDebugView as DebugView, NvofgInterpolator as Interpolator, NvofgMode as Mode,
    NvofgQuality as Quality,
};

// Re-export the feature flags.
pub mod flags {
    pub use nvofg_sys::{
        NVOFG_FLAG_BIDIRECTIONAL as BIDIRECTIONAL, NVOFG_FLAG_HDR as HDR,
        NVOFG_FLAG_USE_DEPTH as USE_DEPTH, NVOFG_FLAG_USE_MATERIAL_ID as USE_MATERIAL_ID,
        NVOFG_FLAG_USE_MOTION as USE_MOTION, NVOFG_FLAG_USE_REACTIVE as USE_REACTIVE,
        NVOFG_FLAG_USE_UI_MASK as USE_UI_MASK,
    };
}

/// Error returned by nvofg (any non-success result except graceful `Unsupported`,
/// which `FrameGen::new` maps to `Ok(None)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidArgument,
    NotRegistered,
    DeviceLost,
    OutOfMemory,
    Internal,
    /// Unexpected: `Unsupported` surfaced somewhere other than `new`.
    Unsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nvofg error: {self:?}")
    }
}
impl std::error::Error for Error {}

fn check(r: sys::NvofgResult) -> Result<(), Error> {
    use sys::NvofgResult as R;
    match r {
        R::Ok => Ok(()),
        R::Unsupported => Err(Error::Unsupported),
        R::InvalidArgument => Err(Error::InvalidArgument),
        R::NotRegistered => Err(Error::NotRegistered),
        R::DeviceLost => Err(Error::DeviceLost),
        R::OutOfMemory => Err(Error::OutOfMemory),
        R::Internal => Err(Error::Internal),
    }
}

/// A registered Vulkan image (a copy of the handle; nvofg does not own it).
#[derive(Debug, Clone, Copy)]
pub struct ImageDesc {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl ImageDesc {
    fn to_sys(&self) -> sys::NvofgImageDesc {
        sys::NvofgImageDesc {
            image: self.image,
            view: self.view,
            format: self.format,
            width: self.width,
            height: self.height,
        }
    }
}

/// Optional auxiliary inputs — each improves quality when present (see the design).
#[derive(Debug, Default, Clone, Copy)]
pub struct Aux {
    pub depth: Option<ImageDesc>,
    pub motion: Option<ImageDesc>,
    pub ui_mask: Option<ImageDesc>,
    pub reactive: Option<ImageDesc>,
    pub material_id: Option<ImageDesc>,
}

/// OFA capabilities (mirrors `NvofgCaps`).
pub type Caps = sys::NvofgCaps;

/// Timeline point to wait on before presenting the interpolated frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameSync {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}

/// Everything needed to create a [`FrameGen`]. Raw handles keep it backend-agnostic.
#[derive(Clone, Copy)]
pub struct CreateInfo {
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    /// Compute-capable queue for prep/refine/warp.
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    /// A queue from the optical-flow family (see [`optical_flow_queue_family`]).
    pub of_queue: vk::Queue,
    pub of_queue_family_index: u32,
    /// `vkGetInstanceProcAddr` from the app's loader (e.g. `entry.static_fn().get_instance_proc_addr`).
    pub get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub extent: vk::Extent2D,
    pub quality: Quality,
    pub interpolator: Interpolator,
    pub mode: Mode,
    pub flags: u32,
}

/// Per-frame generation parameters.
#[derive(Clone, Copy)]
pub struct GenerateInfo {
    /// Position of the generated frame in (0,1); 0.5 = 2×.
    pub phase: f32,
    /// `prevVP_unjittered · inverse(currVP_unjittered)`, row-major (for depth reproj).
    pub reproj: [f32; 16],
    pub near_plane: f32,
    pub far_plane: f32,
    /// Camera cut/teleport: skip interpolation, duplicate a real frame.
    pub reset: bool,
    /// Timeline point the app signals when prev/curr color are render-complete.
    /// `None` if ordering is already guaranteed.
    pub input: Option<FrameSync>,
    pub prev_layout: vk::ImageLayout,
    pub curr_layout: vk::ImageLayout,
}

impl Default for GenerateInfo {
    fn default() -> Self {
        let mut reproj = [0.0f32; 16];
        reproj[0] = 1.0;
        reproj[5] = 1.0;
        reproj[10] = 1.0;
        reproj[15] = 1.0;
        Self {
            phase: 0.5,
            reproj,
            near_plane: 0.0,
            far_plane: 0.0,
            reset: false,
            input: None,
            prev_layout: vk::ImageLayout::GENERAL,
            curr_layout: vk::ImageLayout::GENERAL,
        }
    }
}

/// Find the optical-flow queue family so the app can create a queue from it at
/// device creation. `None` if the device has no OFA.
pub fn optical_flow_queue_family(
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    gipa: vk::PFN_vkGetInstanceProcAddr,
) -> Option<u32> {
    let mut family = 0u32;
    // SAFETY: valid handles + gipa provided by the caller.
    let r = unsafe {
        sys::nvofg_optical_flow_queue_family(instance, physical_device, gipa, &mut family)
    };
    (r == sys::NvofgResult::Ok).then_some(family)
}

/// Query OFA support without committing (for a settings menu). `None` if unusable.
pub fn query_support(
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    gipa: vk::PFN_vkGetInstanceProcAddr,
) -> Option<Caps> {
    let mut caps = Caps::default();
    // SAFETY: valid handles + gipa provided by the caller.
    let r = unsafe { sys::nvofg_query_support(instance, physical_device, gipa, &mut caps) };
    (r == sys::NvofgResult::Ok && caps.supported != 0).then_some(caps)
}

/// The device extensions nvofg needs enabled at device creation.
pub fn required_device_extensions() -> Vec<&'static CStr> {
    let mut count = 0u32;
    // SAFETY: returns a process-lifetime NULL-terminated array of C strings.
    unsafe {
        let ptr = sys::nvofg_required_device_extensions(&mut count);
        (0..count as isize)
            .map(|i| CStr::from_ptr(*ptr.offset(i)))
            .collect()
    }
}

/// A frame generator. RAII: dropping it destroys the underlying context.
pub struct FrameGen {
    ctx: *mut sys::NvofgContext,
}

impl FrameGen {
    /// Create a generator. Returns `Ok(None)` if the GPU/driver has no usable OFA —
    /// the caller then runs at 1×.
    pub fn new(info: &CreateInfo) -> Result<Option<FrameGen>, Error> {
        let ci = sys::NvofgCreateInfo {
            instance: info.instance,
            physical_device: info.physical_device,
            device: info.device,
            queue: info.queue,
            queue_family_index: info.queue_family_index,
            of_queue: info.of_queue,
            of_queue_family_index: info.of_queue_family_index,
            gipa: info.get_instance_proc_addr,
            width: info.extent.width,
            height: info.extent.height,
            quality: info.quality,
            interpolator: info.interpolator,
            mode: info.mode,
            flags: info.flags,
        };
        let mut ctx: *mut sys::NvofgContext = std::ptr::null_mut();
        // SAFETY: `ci` is fully initialised; `ctx` is a valid out-pointer.
        let r = unsafe { sys::nvofg_create(&ci, &mut ctx) };
        match r {
            sys::NvofgResult::Ok => Ok(Some(FrameGen { ctx })),
            sys::NvofgResult::Unsupported => Ok(None),
            other => Err(check(other).unwrap_err()),
        }
    }

    pub fn caps(&self) -> Caps {
        let mut caps = Caps::default();
        // SAFETY: `ctx` is valid for the lifetime of `self`.
        unsafe { sys::nvofg_caps(self.ctx, &mut caps) };
        caps
    }

    pub fn register_color(&mut self, prev: &ImageDesc, curr: &ImageDesc) -> Result<(), Error> {
        // SAFETY: descriptors are valid for the duration of the call.
        check(unsafe { sys::nvofg_register_color(self.ctx, &prev.to_sys(), &curr.to_sys()) })
    }

    pub fn register_aux(&mut self, aux: &Aux) -> Result<(), Error> {
        // Keep the sys descriptors alive across the call.
        let depth = aux.depth.map(|d| d.to_sys());
        let motion = aux.motion.map(|d| d.to_sys());
        let ui = aux.ui_mask.map(|d| d.to_sys());
        let reactive = aux.reactive.map(|d| d.to_sys());
        let material = aux.material_id.map(|d| d.to_sys());
        let opt = |o: &Option<sys::NvofgImageDesc>| {
            o.as_ref().map_or(std::ptr::null(), |d| d as *const _)
        };
        let sys_aux = sys::NvofgAuxDesc {
            depth: opt(&depth),
            motion: opt(&motion),
            ui_mask: opt(&ui),
            reactive: opt(&reactive),
            material_id: opt(&material),
        };
        // SAFETY: all pointers reference locals that outlive the call.
        check(unsafe { sys::nvofg_register_aux(self.ctx, &sys_aux) })
    }

    pub fn register_output(&mut self, interpolated: &ImageDesc) -> Result<(), Error> {
        // SAFETY: descriptor valid for the call.
        check(unsafe { sys::nvofg_register_output(self.ctx, &interpolated.to_sys()) })
    }

    pub fn unregister_all(&mut self) {
        // SAFETY: `ctx` valid.
        unsafe { sys::nvofg_unregister_all(self.ctx) }
    }

    /// Resize to a new present resolution (swapchain recreation). Clears all
    /// registrations — re-register color/aux/output with the new images before the
    /// next [`record_generate`](Self::record_generate).
    pub fn resize(&mut self, extent: vk::Extent2D) -> Result<(), Error> {
        // SAFETY: `ctx` valid.
        check(unsafe { sys::nvofg_resize(self.ctx, extent.width, extent.height) })
    }

    /// Record + submit one generated frame. Returns the timeline point to wait on
    /// before presenting the interpolated image. Does not block the CPU.
    pub fn record_generate(&mut self, gen: &GenerateInfo) -> Result<FrameSync, Error> {
        let (input_sem, input_val) = gen
            .input
            .map_or((vk::Semaphore::null(), 0), |s| (s.semaphore, s.value));
        let gi = sys::NvofgGenerateInfo {
            phase: gen.phase,
            reproj: gen.reproj,
            near_plane: gen.near_plane,
            far_plane: gen.far_plane,
            reset: gen.reset as u32,
            input_timeline: input_sem,
            input_value: input_val,
            prev_layout: gen.prev_layout,
            curr_layout: gen.curr_layout,
            cmd: vk::CommandBuffer::null(),
        };
        let mut out = sys::NvofgFrameSync::default();
        // SAFETY: `gi` fully initialised; `out` valid out-pointer; `ctx` valid.
        check(unsafe { sys::nvofg_record_generate(self.ctx, &gi, &mut out) })?;
        Ok(FrameSync {
            semaphore: out.semaphore,
            value: out.value,
        })
    }

    pub fn set_debug_view(&mut self, view: DebugView, target: Option<&ImageDesc>) -> Result<(), Error> {
        let t = target.map(|d| d.to_sys());
        let tp = t.as_ref().map_or(std::ptr::null(), |d| d as *const _);
        // SAFETY: `tp` references a local that outlives the call (or is null).
        check(unsafe { sys::nvofg_set_debug_view(self.ctx, view, tp) })
    }

    /// Human-readable last error string from the library.
    pub fn last_error(&self) -> String {
        // SAFETY: returns a valid C string owned by the context.
        unsafe { CStr::from_ptr(sys::nvofg_last_error(self.ctx)) }
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for FrameGen {
    fn drop(&mut self) {
        // SAFETY: `ctx` was created by nvofg_create and not yet destroyed.
        unsafe { sys::nvofg_destroy(self.ctx) }
    }
}
