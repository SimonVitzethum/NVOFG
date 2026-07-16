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

if(NOT NVOFG_LIBRARY_ONLY)
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
