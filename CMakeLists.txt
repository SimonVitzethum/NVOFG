cmake_minimum_required(VERSION 3.24)
project(nvofg LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

find_package(Vulkan REQUIRED)

# Warnings on by default; the library core aims to be warning-clean.
add_library(nvofg_warnings INTERFACE)
target_compile_options(nvofg_warnings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wpedantic>)

# ---------------------------------------------------------------------------
# M0 support probe: prove the native VK_NV_optical_flow path on this machine.
# Standalone; depends only on the Vulkan loader. Not linked into the library.
# ---------------------------------------------------------------------------
add_executable(nvofg_probe src/spike/probe.cpp)
target_include_directories(nvofg_probe PRIVATE include)
target_link_libraries(nvofg_probe PRIVATE Vulkan::Vulkan nvofg_warnings)

# The library and Rust crates are wired in as milestones land (M1+).
