# RenderFX — a Linux-first, Vulkan-first modular rendering-effects framework that
# *composes* focused libraries (nvofg for frame generation) behind one API.
#
# Built here via add_subdirectory from the nvofg tree for convenience; it is
# architecturally a SEPARATE project (design.md §13). As a standalone build it would
# find_package(nvofg) instead of using the in-tree `nvofg` target.

add_library(renderfx
  src/backends.cpp
  src/resolve.cpp
  src/context.cpp)
target_include_directories(renderfx
  PUBLIC  include
  PRIVATE ${CMAKE_SOURCE_DIR}/include)   # nvofg.h
target_link_libraries(renderfx PUBLIC Vulkan::Vulkan PRIVATE nvofg)
set_target_properties(renderfx PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)

if(NOT NVOFG_LIBRARY_ONLY)
  add_executable(renderfx_test_resolve tests/test_resolve.cpp)
  target_link_libraries(renderfx_test_resolve PRIVATE renderfx)
  add_test(NAME renderfx_resolve COMMAND renderfx_test_resolve)
endif()
