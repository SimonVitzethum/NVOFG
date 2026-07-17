# Path B — DLSS Frame Generation native-load experiment (design.md §20).
# The DLL under test is NOT vendored: it ships with the NVIDIA driver at
# /usr/lib/nvidia/wine/nvngx_dlssg.dll (Windows PE, NVIDIA-licensed).
DLLSG ?= /usr/lib/nvidia/wine/nvngx_dlssg.dll
CFLAGS ?= -O2 -Wall

pe_probe: pe_probe.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

pe_load: pe_load.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

s5_host: s5_host.c
	$(CC) $(CFLAGS) -o $@ $< -ldl -lvulkan

.PHONY: run recon clean
all: pe_probe pe_load s5_host
run: pe_probe pe_load
	./pe_probe $(DLLSG)
	./pe_load $(DLLSG)
# regenerate the import recon with winedump (needs wine tools)
recon:
	winedump -j import $(DLLSG) || true
clean:
	rm -f pe_probe pe_load s5_host probe_output.txt load_output.txt
