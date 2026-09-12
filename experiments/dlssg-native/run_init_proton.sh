#!/usr/bin/env bash
# Wine-first Init harness: build ngxfg_init_harness.exe (mingw) and run it under
# GE-Proton against the real driver. Ground truth for s5_host's RUN_INIT path:
# if Init returns 0x1 here, our native args are right and the bug is loader-side.
# Usage: ./run_init_proton.sh [0|1]   (FeatureInfo NULL vs &FeatureCommonInfo)
#   WINEDEBUG="+relay" ./run_init_proton.sh 1   (call-sequence capture, see below)
set -e
VARIANT="${1:-1}"
PROTON=~/.local/share/proton/GE-Proton11-1/proton
VKINC=/tmp/vkinc; NGXINC="$(git rev-parse --show-toplevel)/renderfx/vendor/ngx/include"
mkdir -p "$VKINC/vulkan" "$VKINC/vk_video"; cp /usr/include/vulkan/*.h "$VKINC/vulkan/"; cp /usr/include/vk_video/*.h "$VKINC/vk_video/" 2>/dev/null || true
x86_64-w64-mingw32-gcc -O2 -o ngxfg_init_harness.exe ngxfg_init_harness.c -I"$VKINC" -I"$NGXINC"
D=/tmp/ngxtest-init; mkdir -p "$D" /tmp/protonpfx-init; cp ngxfg_init_harness.exe "$D/"
cp /usr/lib/nvidia/wine/nvngx.dll /usr/lib/nvidia/wine/_nvngx.dll /usr/lib/nvidia/wine/nvngx_dlssg.dll "$D/"
cd "$D"; rm -f ngxfg_init_result.txt
STEAM_COMPAT_DATA_PATH=/tmp/protonpfx-init STEAM_COMPAT_CLIENT_INSTALL_PATH=/tmp/protonpfx-init PROTON_ENABLE_NVAPI=1 \
  "$PROTON" run "$D/ngxfg_init_harness.exe" "$VARIANT" >/dev/null 2>&1 || true
cat "$D/ngxfg_init_result.txt"
