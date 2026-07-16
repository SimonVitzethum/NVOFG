# RenderFX — a Linux-first, Vulkan-first modular rendering-effects framework that
# *composes* focused libraries (nvofg for frame generation) behind one API.
#
# Built here via add_subdirectory from the nvofg tree for convenience; it is
# architecturally a SEPARATE project (design.md §13). As a standalone build it would
# find_package(nvofg) instead of using the in-tree `nvofg` target.

# Native upscaling shader (uses the parent tree's Slang -> SPIR-V -> embed helper).
set(RENDERFX_SHADERS "")
nvofg_add_shader(RENDERFX_SHADERS SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/upscale_native.slang
                 ENTRY main SYMBOL rfx_spv_upscale_native)
nvofg_add_shader(RENDERFX_SHADERS SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/upscale_temporal.slang
                 ENTRY main SYMBOL rfx_spv_upscale_temporal)
nvofg_add_shader(RENDERFX_SHADERS SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/rr_debug.slang
                 ENTRY main SYMBOL rfx_spv_rr_debug)
add_custom_target(renderfx_shaders DEPENDS ${RENDERFX_SHADERS})

add_library(renderfx
  src/backends.cpp
  src/resolve.cpp
  src/context.cpp
  src/upscale.cpp
  src/rr_debug.cpp)
add_dependencies(renderfx renderfx_shaders)
target_include_directories(renderfx
  PUBLIC  include
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${NVOFG_SHADER_GEN_DIR})   # nvofg.h + generated SPIR-V
target_link_libraries(renderfx PUBLIC Vulkan::Vulkan PRIVATE nvofg)
set_target_properties(renderfx PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

# -------------------------------------------------------------------------------
# Optional official NVIDIA NGX backend (DLSS SR / DLAA / Ray Reconstruction).
# design.md §16 / ADR 0006: use NGX where available; unsupported otherwise. Needs the
# vendored SDK under vendor/ngx (headers + libnvsdk_ngx.a + rel/ feature blobs).
# -------------------------------------------------------------------------------
option(RENDERFX_NGX "Build the official NVIDIA NGX backend (DLSS SR/DLAA/RR)" OFF)
set(RFX_NGX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ngx")
set(RFX_NGX_LIB "${RFX_NGX_DIR}/lib/Linux_x86_64/libnvsdk_ngx.a")
if(RENDERFX_NGX AND EXISTS "${RFX_NGX_LIB}")
  message(STATUS "renderfx: NGX backend enabled (${RFX_NGX_LIB})")
  target_include_directories(renderfx PRIVATE ${RFX_NGX_DIR}/include)
  target_compile_definitions(renderfx PRIVATE RENDERFX_NGX
    RENDERFX_NGX_DATA_PATH="${RFX_NGX_DIR}/lib/Linux_x86_64/rel")
  target_link_libraries(renderfx PRIVATE ${RFX_NGX_LIB} stdc++ dl)
elseif(RENDERFX_NGX)
  message(WARNING "renderfx: RENDERFX_NGX set but ${RFX_NGX_LIB} not found; NGX disabled")
endif()

if(NOT NVOFG_LIBRARY_ONLY)
  if(RENDERFX_NGX AND EXISTS "${RFX_NGX_LIB}")
    add_executable(renderfx_ngx_probe src/ngx_probe.cpp)
    target_include_directories(renderfx_ngx_probe PRIVATE ${RFX_NGX_DIR}/include)
    target_compile_definitions(renderfx_ngx_probe PRIVATE
      RENDERFX_NGX_DATA_PATH="${RFX_NGX_DIR}/lib/Linux_x86_64/rel")
    target_link_libraries(renderfx_ngx_probe PRIVATE Vulkan::Vulkan ${RFX_NGX_LIB} stdc++ dl)
  endif()

  add_executable(renderfx_test_resolve tests/test_resolve.cpp)
  target_link_libraries(renderfx_test_resolve PRIVATE renderfx)
  add_test(NAME renderfx_resolve COMMAND renderfx_test_resolve)

  add_executable(renderfx_test_upscale tests/test_upscale.cpp)
  target_link_libraries(renderfx_test_upscale PRIVATE renderfx Vulkan::Vulkan)
  add_test(NAME renderfx_upscale COMMAND renderfx_test_upscale)

  add_executable(renderfx_test_rr_debug tests/test_rr_debug.cpp)
  target_link_libraries(renderfx_test_rr_debug PRIVATE renderfx Vulkan::Vulkan)
  add_test(NAME renderfx_rr_debug COMMAND renderfx_test_rr_debug)
endif()
