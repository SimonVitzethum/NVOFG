# Path B — DLSS Frame Generation native-load experiment (design.md §20).
# DLSS-G sources, in preference order (all NVIDIA-licensed, never committed):
#  1. S5_WINE=<dir>  — explicit drop under test (e.g. the vendored 310.9.1 below)
#  2. ../../renderfx/vendor/dlssg/ — in-repo vendored drop (Streamline 2.14.1,
#     nvngx_dlssg.dll 310.9.1.0, git-ignored; see RECON.md for provenance)
#  3. /usr/lib/nvidia/wine/ — the driver's shipped 310.2.1.0 (fallback)
DLLSG ?= /usr/lib/nvidia/wine/nvngx_dlssg.dll
VENDOR_DLSSG = ../../renderfx/vendor/dlssg
CFLAGS ?= -O2 -Wall -g
NGXINC ?= ../../renderfx/vendor/ngx/include

pe_probe: pe_probe.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

pe_load: pe_load.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

s5_host: s5_host.c
	$(CC) $(CFLAGS) -I$(NGXINC) -o $@ $< -ldl -lvulkan -lcrypto

# Full native FG chain gate (needs single-CPU determinism for Init):
# GFR GREEN -> Init 0x1 -> Create 0x1 -> Evaluate (S5_* env documented in s5_host.c).
green-init: s5_host
	S5_SDKVER=0x13 S5_ARG8INT=0x15 S5_SYNC_PHASE=post NVIDIA_WINE_DLL_DIR=/usr/lib/nvidia/wine RUN_INIT=1 taskset -c 0 ./s5_host
green-create: s5_host
	S5_SDKVER=0x15 S5_ARG8INT=0x15 S5_SYNC_PHASE=post NVIDIA_WINE_DLL_DIR=/usr/lib/nvidia/wine RUN_INIT=1 S5_CREATE=1 taskset -c 0 ./s5_host
green-eval: s5_host
	S5_SDKVER=0x15 S5_ARG8INT=0x15 S5_SYNC_PHASE=post NVIDIA_WINE_DLL_DIR=/usr/lib/nvidia/wine RUN_INIT=1 S5_CREATE=1 S5_EVAL=1 taskset -c 0 ./s5_host

.PHONY: run recon clean green green-vendored
all: pe_probe pe_load s5_host
run: pe_probe pe_load
	./pe_probe $(DLLSG)
	./pe_load $(DLLSG)
# Capability gate: native GFR(FrameGeneration) must report GREEN (exit 0).
green: s5_host
	./s5_host
# Same gate against the in-repo vendored 310.9.1 drop (plus driver host DLLs).
green-vendored: s5_host
	@mkdir -p /tmp/wine-vendored
	@ln -sf /usr/lib/nvidia/wine/_nvngx.dll /usr/lib/nvidia/wine/nvngx.dll /tmp/wine-vendored/
	@cp $(VENDOR_DLSSG)/nvngx_dlssg.dll /tmp/wine-vendored/
	S5_WINE=/tmp/wine-vendored/ ./s5_host
# regenerate the import recon with winedump (needs wine tools)
recon:
	winedump -j import $(DLLSG) || true
clean:
	rm -f pe_probe pe_load s5_host probe_output.txt load_output.txt
