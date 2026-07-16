# cmake -P script: embed a SPIR-V binary as a 4-byte-aligned C byte array.
# Args: -DIN=<file.spv> -DOUT=<file.spv.h> -DSYM=<symbol_name>
#
# Emits:
#   alignas(4) static const unsigned char <SYM>[] = { 0x.., ... };
#   static const unsigned int <SYM>_size = sizeof(<SYM>);
# vkShaderModuleCreateInfo.pCode requires uint32 alignment, hence alignas(4).

file(READ "${IN}" hexcontent HEX)
string(REGEX REPLACE "(..)" "0x\\1," bytes "${hexcontent}")
file(WRITE "${OUT}"
  "// Generated from ${IN} — do not edit.\n"
  "alignas(4) static const unsigned char ${SYM}[] = {${bytes}};\n"
  "static const unsigned int ${SYM}_size = sizeof(${SYM});\n")
