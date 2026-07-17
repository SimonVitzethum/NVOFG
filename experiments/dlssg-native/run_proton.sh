#!/usr/bin/env bash
# Gate 3 oracle: run ngxfg_probe.exe under GE-Proton (complete dxvk-nvapi/vkd3d/dxgi stack)
# against the real driver, to read NVSDK_NGX_VULKAN_GetFeatureRequirements(FrameGeneration).
# The NVIDIA driver DLLs (nvngx.dll/_nvngx.dll/nvngx_dlssg.dll) are loaded from the on-disk
# driver dir; they are copied beside the exe at runtime and are NEVER committed.
set -e
PROTON=~/.local/share/proton/GE-Proton11-1/proton
VKINC=/tmp/vkinc; NGXINC="$(git rev-parse --show-toplevel)/renderfx/vendor/ngx/include"
mkdir -p "$VKINC/vulkan" "$VKINC/vk_video"; cp /usr/include/vulkan/*.h "$VKINC/vulkan/"; cp /usr/include/vk_video/*.h "$VKINC/vk_video/" 2>/dev/null || true
x86_64-w64-mingw32-gcc -O2 -o ngxfg_probe.exe ngxfg_probe.c -I"$VKINC" -I"$NGXINC"
D=/tmp/ngxtest; mkdir -p "$D" /tmp/protonpfx; cp ngxfg_probe.exe "$D/"
cp /usr/lib/nvidia/wine/nvngx.dll /usr/lib/nvidia/wine/_nvngx.dll /usr/lib/nvidia/wine/nvngx_dlssg.dll "$D/"
cd "$D"; rm -f ngxfg_result.txt
STEAM_COMPAT_DATA_PATH=/tmp/protonpfx STEAM_COMPAT_CLIENT_INSTALL_PATH=/tmp/protonpfx PROTON_ENABLE_NVAPI=1 \
  "$PROTON" run "$D/ngxfg_probe.exe" >/dev/null 2>&1 || true
cat "$D/ngxfg_result.txt"
