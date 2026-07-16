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

# M0 (full): run one hardware optical-flow execute and dump the flow field.
add_executable(nvofg_flow_dump src/spike/flow_dump.cpp)
target_link_libraries(nvofg_flow_dump PRIVATE Vulkan::Vulkan nvofg_warnings)

# ---------------------------------------------------------------------------
# libnvofg — the frame-generation library (C ABI in include/nvofg.h).
# ---------------------------------------------------------------------------
include(cmake/Shaders.cmake)

set(NVOFG_SHADER_HEADERS "")
nvofg_add_shader(NVOFG_SHADER_HEADERS SOURCE ${CMAKE_SOURCE_DIR}/src/shaders/prep.slang
                 ENTRY main SYMBOL nvofg_spv_prep)
nvofg_add_shader(NVOFG_SHADER_HEADERS SOURCE ${CMAKE_SOURCE_DIR}/src/shaders/refine.slang
                 ENTRY main SYMBOL nvofg_spv_refine)
nvofg_add_shader(NVOFG_SHADER_HEADERS SOURCE ${CMAKE_SOURCE_DIR}/src/shaders/warp.slang
                 ENTRY main SYMBOL nvofg_spv_warp)
nvofg_add_shader(NVOFG_SHADER_HEADERS SOURCE ${CMAKE_SOURCE_DIR}/src/shaders/debug.slang
                 ENTRY main SYMBOL nvofg_spv_debug)
add_custom_target(nvofg_shaders DEPENDS ${NVOFG_SHADER_HEADERS})

add_library(nvofg
  src/context.cpp
  src/pipeline.cpp)
add_dependencies(nvofg nvofg_shaders)
target_include_directories(nvofg
  PUBLIC  include
  PRIVATE src ${NVOFG_SHADER_GEN_DIR})
target_link_libraries(nvofg PUBLIC Vulkan::Vulkan PRIVATE nvofg_warnings)
set_target_properties(nvofg PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(nvofg_test_lifecycle tests/test_lifecycle.cpp)
target_link_libraries(nvofg_test_lifecycle PRIVATE nvofg Vulkan::Vulkan)
add_test(NAME lifecycle COMMAND nvofg_test_lifecycle)

# Headless 2x interpolation example doubles as an integration test.
add_executable(nvofg_headless_interp examples/headless_interp.cpp)
target_link_libraries(nvofg_headless_interp PRIVATE nvofg Vulkan::Vulkan)
add_test(NAME headless_interp COMMAND nvofg_headless_interp)

# The Rust crates (nvofg-sys, nvofg) are wired in at M3.
