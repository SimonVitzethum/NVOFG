// Link nvofg-sys against a prebuilt libnvofg.
//
// Point NVOFG_LIB_DIR at the CMake build directory that contains libnvofg.a
// (default: <repo>/build). Build the C++ library first, e.g.:
//   cmake -S . -B build && cmake --build build
use std::{env, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let default_build = manifest.join("../../build");
    let lib_dir = env::var("NVOFG_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or(default_build);

    println!("cargo:rerun-if-env-changed=NVOFG_LIB_DIR");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=nvofg");
    // libnvofg drives Vulkan directly and uses the C++ runtime.
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
