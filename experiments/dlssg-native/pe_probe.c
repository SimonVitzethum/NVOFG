// Path B — S1 probe: natively map NVIDIA's Windows DLSS Frame Generation snippet
// (nvngx_dlssg.dll, a PE32+ DLL) inside a *native Linux* process, apply base
// relocations, and resolve its import table against the real native drivers
// (libvulkan.so.1 for vulkan-1.dll, libcuda.so.1 for nvcuda.dll). It then reports the
// exact remaining Win32 (KERNEL32/ADVAPI32/USER32/VERSION) stub surface that a minimal
// CRT shim (S2) must provide. It deliberately does NOT execute the DLL (no DllMain / TLS
// callbacks) — this is a mapping + gap-analysis milestone, not a run.
//
// Rationale (design.md §20): recon showed the FG snippet imports only Vulkan + CUDA +
// MSVC-CRT (no D3D12/DXGI/COM), so most of its surface is satisfiable by native drivers.
// This probe verifies that empirically and quantifies what is left to stub.
//
// Build:  gcc -O2 -o pe_probe pe_probe.c -ldl
// Usage:  ./pe_probe /usr/lib/nvidia/wine/nvngx_dlssg.dll
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;

static u32 rd32(const u8* p) { u32 v; memcpy(&v, p, 4); return v; }
static u16 rd16(const u8* p) { u16 v; memcpy(&v, p, 2); return v; }
static u64 rd64(const u8* p) { u64 v; memcpy(&v, p, 8); return v; }

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/usr/lib/nvidia/wine/nvngx_dlssg.dll";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; fstat(fd, &st);
    u8* file = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file == MAP_FAILED) { perror("mmap file"); return 2; }
    printf("== Path B / S1 PE probe: %s (%ld bytes)\n", path, (long)st.st_size);

    if (rd16(file) != 0x5A4D) { fprintf(stderr, "not MZ\n"); return 2; }      // "MZ"
    u32 e_lfanew = rd32(file + 0x3C);
    const u8* nt = file + e_lfanew;
    if (rd32(nt) != 0x00004550) { fprintf(stderr, "not PE\\0\\0\n"); return 2; } // "PE\0\0"
    const u8* fh = nt + 4;                                   // IMAGE_FILE_HEADER
    u16 machine = rd16(fh + 0);
    u16 nsec = rd16(fh + 2);
    u16 optsz = rd16(fh + 16);
    const u8* oh = fh + 20;                                  // IMAGE_OPTIONAL_HEADER64
    if (rd16(oh) != 0x20B) { fprintf(stderr, "not PE32+\n"); return 2; }
    u64 image_base = rd64(oh + 24);
    u32 size_image = rd32(oh + 56);
    u32 size_hdrs  = rd32(oh + 60);
    u32 ndir       = rd32(oh + 108);
    const u8* dir  = oh + 112;                               // IMAGE_DATA_DIRECTORY[]
    u32 imp_rva  = ndir > 1 ? rd32(dir + 1*8) : 0;
    u32 reloc_rva = ndir > 5 ? rd32(dir + 5*8) : 0;
    u32 reloc_sz  = ndir > 5 ? rd32(dir + 5*8 + 4) : 0;
    printf("   machine=0x%04x sections=%u imageBase=0x%llx sizeOfImage=%u dirs=%u\n",
           machine, nsec, (unsigned long long)image_base, size_image, ndir);

    // --- map the image at an anonymous base; copy headers + sections to their RVAs ---
    u8* base = mmap(NULL, size_image, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap image"); return 2; }
    memcpy(base, file, size_hdrs);
    const u8* sec = oh + optsz;                              // section table
    for (u16 i = 0; i < nsec; ++i) {
        const u8* s = sec + i * 40;
        u32 vsize = rd32(s + 8), vaddr = rd32(s + 12), rawsz = rd32(s + 16), rawptr = rd32(s + 20);
        u32 n = rawsz < vsize ? rawsz : vsize;
        if (vaddr + n <= size_image && rawptr + n <= st.st_size) memcpy(base + vaddr, file + rawptr, n);
    }

    // --- apply base relocations for delta = actual - preferred ---
    int64_t delta = (int64_t)((u64)base - image_base);
    u32 relocs = 0;
    if (reloc_rva && delta) {
        const u8* p = base + reloc_rva; const u8* end = p + reloc_sz;
        while (p + 8 <= end) {
            u32 page = rd32(p), blk = rd32(p + 4);
            if (blk < 8) break;
            u32 cnt = (blk - 8) / 2;
            for (u32 i = 0; i < cnt; ++i) {
                u16 e = rd16(p + 8 + i * 2);
                u32 type = e >> 12, off = e & 0xFFF;
                if (type == 10) { // IMAGE_REL_BASED_DIR64
                    u8* t = base + page + off; u64 v = rd64(t); v += delta; memcpy(t, &v, 8); relocs++;
                }
            }
            p += blk;
        }
    }
    printf("   mapped @ %p  delta=0x%llx  relocations applied=%u\n",
           (void*)base, (unsigned long long)delta, relocs);

    // --- native driver handles for import resolution ---
    void* hvk  = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_GLOBAL);
    void* hcu  = dlopen("libcuda.so.1",   RTLD_NOW | RTLD_GLOBAL);
    printf("   native: libvulkan.so.1=%s  libcuda.so.1=%s\n", hvk ? "ok" : "MISSING", hcu ? "ok" : "MISSING");

    // --- walk the import directory; resolve & report gaps ---
    int nat_vk_ok = 0, nat_vk_miss = 0, nat_cu_ok = 0, nat_cu_miss = 0, win_stub = 0, ordinals = 0;
    char win[4096]; win[0] = 0; char vkmiss[1024]; vkmiss[0] = 0; char cumiss[1024]; cumiss[0] = 0;
    if (imp_rva) {
        const u8* d = base + imp_rva;
        for (;; d += 20) {
            u32 oft = rd32(d + 0), nameRva = rd32(d + 12), ft = rd32(d + 16);
            if (!nameRva && !ft && !oft) break;
            const char* dll = (const char*)(base + nameRva);
            int is_vk = strcasecmp(dll, "vulkan-1.dll") == 0;
            int is_cu = strcasecmp(dll, "nvcuda.dll") == 0;
            u32 thunks = oft ? oft : ft;                    // INT if present, else IAT
            const u8* t = base + thunks;
            u8* iat = base + ft;                            // where to write resolved ptrs
            for (;; t += 8, iat += 8) {
                u64 v = rd64(t);
                if (!v) break;
                if (v & 0x8000000000000000ULL) { ordinals++; continue; }   // import by ordinal
                const char* fn = (const char*)(base + (u32)v + 2);         // skip Hint(2)
                if (is_vk || is_cu) {
                    void* sym = dlsym(is_vk ? hvk : hcu, fn);
                    if (sym) { u64 p = (u64)sym; memcpy(iat, &p, 8); if (is_vk) nat_vk_ok++; else nat_cu_ok++; }
                    else {
                        if (is_vk) { nat_vk_miss++; if (strlen(vkmiss) < 900) { strcat(vkmiss, fn); strcat(vkmiss, " "); } }
                        else       { nat_cu_miss++; if (strlen(cumiss) < 900) { strcat(cumiss, fn); strcat(cumiss, " "); } }
                    }
                } else {
                    win_stub++;
                    if (strlen(win) < 3900) { strcat(win, dll); strcat(win, ":"); strcat(win, fn); strcat(win, "\n"); }
                }
            }
        }
    }

    printf("\n== Import resolution against native drivers ==\n");
    printf("   Vulkan (vulkan-1.dll):  resolved %d, missing %d\n", nat_vk_ok, nat_vk_miss);
    if (nat_vk_miss) printf("      missing: %s\n", vkmiss);
    printf("   CUDA   (nvcuda.dll):    resolved %d, missing %d\n", nat_cu_ok, nat_cu_miss);
    if (nat_cu_miss) printf("      missing: %s\n", cumiss);
    printf("   Win32 CRT stubs needed: %d  (imports by ordinal: %d)\n", win_stub, ordinals);
    printf("\n== S2 stub worklist (Win32 symbols to shim) ==\n%s", win);
    printf("\n== Verdict ==\n");
    printf("   PE mapped + relocated natively: YES (%u relocs).\n", relocs);
    printf("   Vulkan+CUDA imports satisfiable by native drivers: %d/%d.\n",
           nat_vk_ok + nat_cu_ok, nat_vk_ok + nat_vk_miss + nat_cu_ok + nat_cu_miss);
    printf("   Remaining to run DllMain (S2): %d Win32 CRT symbols above + a PE-aware CRT/SEH init.\n", win_stub);
    return 0;
}
