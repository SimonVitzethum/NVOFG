# Slang -> SPIR-V -> embedded C header pipeline.
#
# nvofg_add_shader(<out_var> SOURCE <file.slang> ENTRY <entry> SYMBOL <c_symbol>)
# appends the generated header path to <out_var>. The header defines
# `<c_symbol>[]` / `<c_symbol>_size`. Include it from C++ and hand the bytes to
# vkCreateShaderModule.
#
# slangc is located via SLANGC_EXECUTABLE (cache var, overridable), else PATH,
# else known vendored locations. nvofg does not depend on any specific install.

find_program(SLANGC_EXECUTABLE
  NAMES slangc
  HINTS
    ENV SLANGC
    "$ENV{VULKAN_SDK}/bin"
    "${CMAKE_SOURCE_DIR}/../.RustMineClient/vendor/slang/bin"
  DOC "Path to the Slang compiler (slangc)")

if(NOT SLANGC_EXECUTABLE)
  message(FATAL_ERROR
    "slangc not found. Install Slang or set -DSLANGC_EXECUTABLE=/path/to/slangc.")
endif()
message(STATUS "nvofg: using slangc at ${SLANGC_EXECUTABLE}")

set(NVOFG_SHADER_GEN_DIR "${CMAKE_BINARY_DIR}/generated/shaders")
file(MAKE_DIRECTORY "${NVOFG_SHADER_GEN_DIR}")

function(nvofg_add_shader out_var)
  cmake_parse_arguments(S "" "SOURCE;ENTRY;SYMBOL" "" ${ARGN})
  get_filename_component(_name "${S_SOURCE}" NAME_WE)
  set(_spv "${NVOFG_SHADER_GEN_DIR}/${_name}_${S_ENTRY}.spv")
  set(_hdr "${NVOFG_SHADER_GEN_DIR}/${S_SYMBOL}.spv.h")

  add_custom_command(
    OUTPUT "${_spv}"
    COMMAND "${SLANGC_EXECUTABLE}" "${S_SOURCE}"
            -target spirv -profile glsl_450 -entry "${S_ENTRY}"
            -matrix-layout-row-major -O2 -o "${_spv}"
    DEPENDS "${S_SOURCE}"
    COMMENT "slangc ${_name}:${S_ENTRY} -> SPIR-V"
    VERBATIM)

  add_custom_command(
    OUTPUT "${_hdr}"
    COMMAND ${CMAKE_COMMAND} -DIN=${_spv} -DOUT=${_hdr} -DSYM=${S_SYMBOL}
            -P "${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake"
    DEPENDS "${_spv}" "${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake"
    COMMENT "embed ${S_SYMBOL}"
    VERBATIM)

  set(${out_var} ${${out_var}} "${_hdr}" PARENT_SCOPE)
endfunction()
