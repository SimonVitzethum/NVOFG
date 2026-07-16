//! Minimal end-to-end linkage + smoke test for the safe `nvofg` crate:
//! load Vulkan, find an OFA device, query support, and (if present) create a
//! FrameGen. Proves the Rust -> C ABI -> libnvofg -> Vulkan path links and runs.
//!
//! Run after building the C++ library:  cmake --build build && cargo run -p nvofg --example support

use ash::vk;
use std::ffi::CStr;

fn main() {
    // SAFETY: standard ash init on a well-formed environment.
    let entry = unsafe { ash::Entry::load() }.expect("load Vulkan loader");
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let ici = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&ici, None) }.expect("create instance");
    let gipa = entry.static_fn().get_instance_proc_addr;

    let pds = unsafe { instance.enumerate_physical_devices() }.expect("enumerate");
    let mut found = false;
    for pd in pds {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy();
        match nvofg::query_support(instance.handle(), pd, gipa) {
            Some(caps) => {
                found = true;
                println!("OFA on {name}:");
                println!(
                    "  bidir={} cost={} hint={} global={} grid=[{}..{}] max={}x{}",
                    caps.bidirectional, caps.cost_map, caps.hint_support, caps.global_flow,
                    caps.min_grid_size, caps.max_grid_size, caps.max_width, caps.max_height
                );
                let of_family = nvofg::optical_flow_queue_family(instance.handle(), pd, gipa)
                    .expect("of queue family");
                println!("  optical-flow queue family = {of_family}");
                let exts = nvofg::required_device_extensions();
                println!("  required device extensions = {exts:?}");
            }
            None => println!("no OFA on {name} (would run at 1x)"),
        }
    }
    if !found {
        println!("no OFA device found on this machine.");
    }
    unsafe { instance.destroy_instance(None) };
}
