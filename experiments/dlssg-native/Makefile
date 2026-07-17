# Path B — DLSS Frame Generation native-load experiment (design.md §20).
# The DLL under test is NOT vendored: it ships with the NVIDIA driver at
# /usr/lib/nvidia/wine/nvngx_dlssg.dll (Windows PE, NVIDIA-licensed).
DLLSG ?= /usr/lib/nvidia/wine/nvngx_dlssg.dll
CFLAGS ?= -O2 -Wall

pe_probe: pe_probe.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

.PHONY: run recon clean
run: pe_probe
	./pe_probe $(DLLSG)
# regenerate the import recon with winedump (needs wine tools)
recon:
	winedump -j import $(DLLSG) || true
clean:
	rm -f pe_probe probe_output.txt
