// Path B — S5(a): load the Windows NGX host (_nvngx.dll) natively and reach the
// NVSDK_NGX_VULKAN_* API that can drive the FrameGeneration snippet. Generalises the S2
// loader (pe_load.c) into a multi-PE loader with a module registry + real
// LoadLibrary/GetProcAddress + export-table parsing, so the host can pull its siblings.
//
// Result: the Windows NGX host loads + inits natively (DllMain->1), its NVSDK_NGX_VULKAN
// API resolves, and NVSDK_NGX_VULKAN_Init_ProjectID — called here with a live VkInstance/
// VkDevice + ms_abi->SysV Vulkan/CUDA thunks + ms_abi gipa/gdpa — RUNS to completion and
// returns a clean NGX code 0xBAD00002 (FAIL_PlatformError): it reached GPU-arch detection
// and failed only because nvapi64.dll is not yet bridged. No crash, no ABI issue. The sole
// remaining dependency for a successful Init is an nvapi shim (the dxvk-nvapi role).
// Runs in a forked child under a SIGSEGV guard; loads the on-disk driver DLLs in place.
//
// Build: gcc -O2 -o s5_host s5_host.c -ldl -lvulkan   Run: ./s5_host
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <malloc.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

// Generic Microsoft-x64 -> System-V ABI thunk: the PE calls its Vulkan/CUDA imports
// MS-x64 (rcx,rdx,r8,r9 + shadow); native libvulkan/libcuda are SysV (rdi,rsi,rdx,rcx,
// r8,r9). Target native fn is passed in r10 by the trampoline.
//
// SCOPE / TODO (S6): this handles ONLY the INTEGER class (RCX/RDX/R8/R9 -> RDI/RSI/RDX/RCX)
// + 2 stack-overflow args + shadow space. It is CORRECT for pointer/handle signatures
// (all of Vulkan, and the CUDA driver calls NGX-Init uses) but NOT for float/double args
// or by-value structs: MS-x64 uses positional slots (a float at position 3 -> XMM2) while
// SysV has a separate SSE file (XMM0..7) and classifies structs (INTEGER/SSE/MEMORY)
// independently. When the CUDA-interop path (S6) goes live, any import taking a float/
// double or a small struct by value needs XMM handling + struct classification added here.
__asm__(
".text\n.globl ms2sysv_common\nms2sysv_common:\n"
"  push %rdi\n  push %rsi\n"
"  mov %rcx, %rdi\n"          // sysv arg1 = ms arg1
"  mov %rdx, %rsi\n"          // sysv arg2 = ms arg2
"  mov %r9,  %rax\n"          // save ms arg4 before clobbering rcx
"  mov %r8,  %rdx\n"          // sysv arg3 = ms arg3
"  mov %rax, %rcx\n"          // sysv arg4 = ms arg4
"  mov 0x38(%rsp), %r8\n"     // sysv arg5 = ms stack arg5
"  mov 0x40(%rsp), %r9\n"     // sysv arg6 = ms stack arg6
"  sub $8, %rsp\n"
"  call *%r10\n"
"  add $8, %rsp\n"
"  pop %rsi\n  pop %rdi\n  ret\n");
extern void ms2sysv_common(void);

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
#define MSABI __attribute__((ms_abi))
#define WINE "/usr/lib/nvidia/wine/"
static u32 rd32(const u8* p){u32 v;memcpy(&v,p,4);return v;}
static u16 rd16(const u8* p){u16 v;memcpy(&v,p,2);return v;}
static u64 rd64(const u8* p){u64 v;memcpy(&v,p,8);return v;}
static void logs(const char* s){ (void)write(2,s,strlen(s)); }
static void logn(const char* a,const char* b){ logs(a); logs(b); logs("\n"); }

// ---- module registry ----
typedef struct { char name[64]; u8* base; u64 imgbase; u32 exp_rva,exp_sz; int loaded; } Module;
static Module g_mod[16]; static int g_nmod=0;
static u8* g_code; static size_t g_codeoff;
static __thread void* g_tls[2048]; static __thread int g_lasterr; static int g_tlsnext=1; static u8 g_heap[1];
static u64 g_hcount=0x2000;
static u8 g_teb[0x2000],g_peb[0x800]; static void* g_tlsslots[512];   // fake Windows TEB + PE-TLS array
static void* make_trap(const char* name);   // fwd (defined below)
static void* make_ms2sysv(void* target);     // fwd (defined below)
static u8 g_luid[8];                          // synthetic adapter LUID (deviceUUID-derived; filled in GPDP2)
MSABI static void* my_gipa(void*,const char*);   // ms_abi vkGetInstanceProcAddr wrapper (below)
MSABI static void* my_gdpa(void*,const char*);   // ms_abi vkGetDeviceProcAddr wrapper (below)
static void* g_hvk,*g_hcu;                    // native libvulkan / libcuda (defined below)

// ---- ms_abi CRT shim (subset proven in pe_load.c) ----
MSABI static void* s_GetProcessHeap(void){return g_heap;}
MSABI static void* s_HeapAlloc(void* h,u32 f,u64 s){(void)h;return(f&8)?calloc(1,s):malloc(s);}
MSABI static void* s_HeapReAlloc(void* h,u32 f,void* p,u64 s){(void)h;(void)f;return realloc(p,s);}
MSABI static int   s_HeapFree(void* h,u32 f,void* p){(void)h;(void)f;free(p);return 1;}
MSABI static u64   s_HeapSize(void* h,u32 f,void* p){(void)h;(void)f;return p?malloc_usable_size(p):0;}
MSABI static void* s_HeapCreate(u32 a,u64 b,u64 c){(void)a;(void)b;(void)c;return g_heap;}
MSABI static int   s_HeapDestroy(void* h){(void)h;return 1;}
MSABI static void* s_LocalAlloc(u32 f,u64 s){return(f&0x40)?calloc(1,s):malloc(s);}
MSABI static void* s_LocalFree(void* p){free(p);return 0;}
MSABI static u32   s_TlsAlloc(void){int i=__sync_fetch_and_add(&g_tlsnext,1);return i<2048?i:0xFFFFFFFF;}
MSABI static void* s_TlsGetValue(u32 i){g_lasterr=0;return i<2048?g_tls[i]:0;}
MSABI static int   s_TlsSetValue(u32 i,void* v){if(i<2048){g_tls[i]=v;return 1;}return 0;}
MSABI static int   s_TlsFree(u32 i){(void)i;return 1;}
MSABI static u32   s_GetCurrentThreadId(void){return(u32)(u64)pthread_self();}
MSABI static u32   s_GetCurrentProcessId(void){return(u32)getpid();}
MSABI static void* s_GetCurrentProcess(void){return(void*)-1;}
MSABI static void* s_GetCurrentThread(void){return(void*)-2;}
MSABI static void  s_GetSystemTimeAsFileTime(void* ft){struct timespec t;clock_gettime(CLOCK_REALTIME,&t);u64 v=(u64)t.tv_sec*10000000ULL+t.tv_nsec/100+116444736000000000ULL;memcpy(ft,&v,8);}
MSABI static int   s_QueryPerformanceCounter(void* x){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);u64 v=(u64)t.tv_sec*1000000000ULL+t.tv_nsec;memcpy(x,&v,8);return 1;}
MSABI static int   s_QueryPerformanceFrequency(void* x){u64 v=1000000000ULL;memcpy(x,&v,8);return 1;}
MSABI static void* s_EncodePointer(void* p){return p;}
MSABI static void* s_DecodePointer(void* p){return p;}
MSABI static int   s_IsProcessorFeaturePresent(u32 f){(void)f;return 1;}
MSABI static int   s_IsDebuggerPresent(void){return 0;}
MSABI static void  s_InitializeSListHead(void* h){memset(h,0,16);}
MSABI static void* s_ret0p(void){return 0;}
MSABI static int   s_ret1(void){return 1;}
MSABI static void  s_noop(void){}
MSABI static u32   s_GetLastError(void){return(u32)g_lasterr;}
MSABI static void  s_SetLastError(u32 e){g_lasterr=(int)e;}
MSABI static void* s_CreateHandle(void){return(void*)__sync_fetch_and_add(&g_hcount,1);}
MSABI static int   s_CloseHandle(void* h){(void)h;return 1;}
MSABI static u32   s_WaitForSingleObject(void* h,u32 m){(void)h;(void)m;return 0;}
MSABI static u64   s_GetTickCount64(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return(u64)t.tv_sec*1000+t.tv_nsec/1000000;}
MSABI static void  s_GetStartupInfoW(void* si){if(si)memset(si,0,104);}
MSABI static void* s_GetCommandLineW(void){static const u16 w[]={'a',0};return(void*)w;}
MSABI static void  s_OutputDebugStringA(const char* s){logn("[dbg] ",s?s:"");}
static u16 g_env[2]={0,0};
MSABI static void* s_GetEnvironmentStringsW(void){return g_env;}
MSABI static int   s_FreeEnvironmentStringsW(void* p){(void)p;return 1;}
MSABI static u32   s_GetEnvironmentVariableA(const char* n,void* b,u32 s){(void)n;(void)b;(void)s;g_lasterr=203;return 0;}
MSABI static int s_WideCharToMultiByte(u32 cp,u32 fl,const u16* w,int wl,char* mb,int mbl,void* d,void* u){(void)cp;(void)fl;(void)d;(void)u;int i=0;if(wl<0){int n=0;while(w[n])n++;wl=n+1;}if(mbl==0)return wl;for(;i<wl&&i<mbl;i++)mb[i]=(char)(w[i]&0xFF);return i;}
MSABI static int s_MultiByteToWideChar(u32 cp,u32 fl,const char* mb,int mbl,u16* w,int wl){(void)cp;(void)fl;int i=0;if(mbl<0){int n=0;while(mb[n])n++;mbl=n+1;}if(wl==0)return mbl;for(;i<mbl&&i<wl;i++)w[i]=(u8)mb[i];return i;}
MSABI static int s_InitOnceExecuteOnce(void* o,void* f,void* p,void** c){(void)p;(void)c;typedef int MSABI(*t)(void*,void*,void**);if(f)((t)f)(o,p,c);return 1;}
static u32 putw16(u16* b,u32 sz,const char* s){ u32 n=strlen(s); if(!b||sz==0) return n; u32 i=0; for(;i<n&&i<sz-1;i++) b[i]=(u8)s[i]; b[i]=0; return i; }
MSABI static u32 s_GetSystemDirectoryW(u16* b,u32 sz){ return putw16(b,sz,"C:\\Windows\\System32"); }
MSABI static u32 s_GetWindowsDirectoryW(u16* b,u32 sz){ return putw16(b,sz,"C:\\Windows"); }
MSABI static u32 s_GetModuleFileNameW(void* m,u16* b,u32 sz){ (void)m; return putw16(b,sz,"C:\\game\\game.exe"); }
MSABI static u32 s_GetModuleFileNameA(void* m,char* b,u32 sz){ (void)m; const char* s="C:\\game\\game.exe"; u32 n=strlen(s); if(!b||!sz)return n; u32 i=0; for(;i<n&&i<sz-1;i++)b[i]=s[i]; b[i]=0; return i; }
MSABI static int s_VerifyVersionInfoW(void* a,u32 b,u64 c){ (void)a;(void)b;(void)c; return 1; }
MSABI static u64 s_VerSetConditionMask(u64 m,u32 t,u8 c){ (void)t;(void)c; return m?m:1; }
MSABI static u32 s_GetFullPathNameW(const u16* n,u32 sz,u16* buf,void** fp){ (void)fp; u32 i=0; if(buf)for(;n[i]&&i<sz-1;i++)buf[i]=n[i]; if(buf)buf[i]=0; return i; }
// --- S5e file-API trace: log the exact paths NGX asks for during NGXGetPath, so we can tell
// whether it wants a WRITABLE DIRECTORY (scenario A) or concrete MODEL FILES (scenario B). ---
static void wlog(const char* tag,const void* w){ char b[320]; const u16* p=w; int i=0; if(p) for(;p[i]&&i<319;i++) b[i]=(char)(p[i]&0xFF); b[i]=0; logn(tag,b); }
MSABI static void* s_CreateFileW(void* name,u32 a,u32 s,void* sa,u32 cd,u32 fa,void* t){ (void)a;(void)s;(void)sa;(void)cd;(void)fa;(void)t; wlog("[CreateFileW] ",name); g_lasterr=2/*ERROR_FILE_NOT_FOUND*/; return (void*)-1; } // INVALID_HANDLE_VALUE
MSABI static u32   s_GetFileAttributesW(void* name){ wlog("[GetFileAttributesW] ",name); g_lasterr=2/*ERROR_FILE_NOT_FOUND*/; return 0xFFFFFFFF; } // INVALID_FILE_ATTRIBUTES
MSABI static int   s_GetFileAttributesExW(void* name,u32 lvl,void* info){ (void)lvl;(void)info; wlog("[GetFileAttributesExW] ",name); return 0; } // FALSE
MSABI static void* s_FindFirstFileExW(void* name,u32 a,void* d,u32 b,void* c,u32 e){ (void)a;(void)d;(void)b;(void)c;(void)e; wlog("[FindFirstFileExW] ",name); return (void*)-1; }
// NGX's NGXGetPath calls SHGetKnownFolderPath(rfid, flags, token, &out) to locate its
// model/config dir (ProgramData/LocalAppData). Return a real writable path (the §19 pattern
// from the Win32 side). Caller frees with CoTaskMemFree -> stub it. HRESULT S_OK = 0.
MSABI static int s_SHGetKnownFolderPath(void* rfid,u32 fl,void* tok,u16** out){ (void)rfid;(void)fl;(void)tok;
    const char* path="C:\\ProgramData"; u16* w=malloc(64*2); int i=0; for(;path[i];i++) w[i]=(u8)path[i]; w[i]=0;
    if(out)*out=w; logn("[SHGetKnownFolderPath] -> ",path); return 0; }
MSABI static void s_CoTaskMemFree(void* p){ free(p); }
// D3DKMTEnumAdapters2: NGX's OS display-adapter enumeration (this is where the OS-adapter LUID
// comes from, to correlate with the GPU). Report ONE adapter whose LUID == our synthetic
// deviceUUID-derived LUID (g_luid), matching the Vulkan deviceLUID. Under Proton this is Wine's
// win32u; natively we synthesize it. STATUS_SUCCESS=0.
// D3DKMT_ENUMADAPTERS2{ ULONG NumAdapters@0; D3DKMT_ADAPTERINFO* pAdapters@8 }
// D3DKMT_ADAPTERINFO{ UINT hAdapter@0; LUID AdapterLuid@4; ULONG NumOfSources@12; BOOL @16 } (20B)
MSABI static int s_D3DKMTEnumAdapters2(void* pe){ if(!pe) return (int)0xC000000D /*STATUS_INVALID_PARAMETER*/;
    u32* n=(u32*)pe; void** pa=(void**)((u8*)pe+8);
    if(!*pa){ *n=1; return 0; }                       // query: report count only
    u8* a=(u8*)*pa; memset(a,0,20); *(u32*)(a+0)=0x21; memcpy(a+4,g_luid,8); *(u32*)(a+12)=1; *n=1;
    logn("[D3DKMTEnumAdapters2] ","1 adapter, LUID=deviceUUID-derived"); return 0; }
MSABI static int s_PathFileExistsW(void* p){ (void)p; return 0; } // FALSE (deny_list etc. absent = fine)
// D3DKMT_QUERYADAPTERINFO{ UINT hAdapter@0; KMTQAITYPE Type@4; void* pData@8; UINT Size@16 }
MSABI static int s_D3DKMTQueryAdapterInfo(void* p){ if(!p) return (int)0xC000000D;
    u32 type=*(u32*)((u8*)p+4); void* pd=*(void**)((u8*)p+8); u32 sz=*(u32*)((u8*)p+16);
    char b[80]; snprintf(b,sizeof b,"[D3DKMTQueryAdapterInfo] Type=%u size=%u\n",type,sz); logs(b);
    if(pd&&sz) memset(pd,0,sz);
    // Type 48 = KMTQAITYPE_KMD_DRIVER_VERSION: NGX checks the WDDM version >= a minimum for FG.
    // Report a modern WDDM (KMT_DRIVERVERSION_WDDM_3_0 = 3000). Zeroed = version 0 = too old.
    if(type==48 && pd && sz>=4) *(u32*)pd = 3000;
    return 0; } // STATUS_SUCCESS
MSABI static int s_D3DKMTCloseAdapter(void* p){ (void)p; return 0; }
MSABI static int s_D3DKMTOpenAdapterFromLuid(void* p){ if(p) *(u32*)p=0x21; return 0; } // fill hAdapter
// --- nvapi shim (S5c): the host resolves NvAPI_* via nvapi_QueryInterface(id). First pass:
// log every requested interface id and hand back a 0-returning stub (NVAPI_OK) so the host
// keeps querying and we can enumerate the full set it needs for GPU-arch detection. ---
static void* g_hnvml;   // native libnvidia-ml.so.1
// nvapi shim (S5c, RE'd via NVIDIA/nvapi + dxvk-nvapi): NGX resolves NvAPI_Initialize +
// the DRS (driver-settings) functions to read DLSS overrides. dxvk-nvapi implements exactly
// these and NOT the 3 private IDs NGX also probes (0xAD298D3F/0x33C7358C/0x593E8644) — yet
// DLSS-G works under Proton, so those are optional. We provide Init + DRS as no-op successes
// (empty settings -> NGX uses defaults) and NULL for everything else; GPU arch comes from
// Vulkan (native). NVAPI_OK = 0.
MSABI static int s_NvAPI_Initialize(void){ return 0; }
MSABI static int s_NvAPI_Unload(void){ return 0; }
MSABI static int s_DRS_CreateSession(void** ph){ if(ph)*ph=(void*)0x1D25; return 0; }
MSABI static int s_DRS_LoadSettings(void* h){ (void)h; return 0; }
MSABI static int s_DRS_DestroySession(void* h){ (void)h; return 0; }
MSABI static int s_DRS_GetBaseProfile(void* h,void** ph){ (void)h; if(ph)*ph=(void*)0x1B45E; return 0; }
MSABI static int s_DRS_GetSetting(void* h,void* p,u32 id,void* out){ (void)h;(void)p;(void)id;(void)out; return -160; } // NVAPI_SETTING_NOT_FOUND (nvapi status, NOT an NGX code) -> NGX uses default
MSABI static int s_DRS_FindApplicationByName(void* sess,void* appName,void** phProfile,void* pApp){ (void)sess;(void)appName;(void)pApp; if(phProfile)*phProfile=(void*)0x1B45E; return 0; } // dxvk-nvapi returns Ok
// NVDRS_PROFILE = {version@0, profileName[2048]@4 (NvU16 inline), gpuSupport, isPredefined, numOfApps, numOfSettings}.
// dxvk-nvapi zeroes profileName (=> valid empty wstring, not NULL) + the trailing fields. That's what NGX wcslen's.
MSABI static int s_DRS_GetProfileInfo(void* sess,void* prof,u8* p){ (void)sess;(void)prof; if(!p) return -5 /*NVAPI_INVALID_ARGUMENT*/; memset(p+4,0,4096+16); return 0; }
MSABI static int s_SYS_GetDriverAndBranchVersion(u32* pver,char* branch){ if(pver)*pver=61043; /*610.43*/ if(branch){const char* b="r610_00";int i=0;for(;b[i]&&i<63;i++)branch[i]=b[i];branch[i]=0;} return 0; }
// Back the fake nvapi handles with REAL zeroed memory, so if NGX dereferences a handle
// (the Windows nvapi handles are internally pointers) it lands in valid memory, not a crash.
static u8 g_gpubuf[8192];
#define FAKE_GPU  ((void*)(g_gpubuf))
#define FAKE_LGPU ((void*)(g_gpubuf+4096))
MSABI static int s_EnumPhysicalGPUs(void** h,u32* c){ if(h)h[0]=FAKE_GPU; if(c)*c=1; return 0; }
// NV_GPU_ARCH_INFO = {version@0 (set by caller), architecture@4, implementation@8, revision@12}.
// We know the device is a Blackwell RTX 5070 -> report GB200 (0x1B0), which is >= Ada, so FG-eligible.
MSABI static int s_GPU_GetArchInfo(void* gpu,u32* ai){ (void)gpu; if(ai){ ai[1]=0x000001B0; /*GB200/Blackwell*/ ai[2]=0x00000005; /*GB20x impl*/ ai[3]=0x000000A1; /*rev*/ } return 0; }
MSABI static int s_GetLogicalGPU(void* p,void** l){ (void)p; if(l)*l=FAKE_LGPU; return 0; }
MSABI static int s_GPU_GetPCIIdentifiers(void* g,u32* dev,u32* sub,u32* rev,u32* ext){ (void)g; u32 id=(0x2D18u<<16)|0x10DE; if(dev)*dev=id; if(sub)*sub=0; if(rev)*rev=0xA1; if(ext)*ext=id; return 0; }
MSABI static int s_GPU_GetFullName(void* g,char* name){ (void)g; const char* s="NVIDIA GeForce RTX 5070 Laptop GPU"; if(name){int i=0;for(;s[i]&&i<63;i++)name[i]=s[i];name[i]=0;} return 0; }
MSABI static int s_GPU_GetGPUType(void* g,u32* t){ (void)g; if(t)*t=2; /*DGPU*/ return 0; }
MSABI static int s_GPU_GetBusType(void* g,u32* t){ (void)g; if(t)*t=3; /*PCI_EXPRESS*/ return 0; }
// Synthetic Windows adapter LUID, used for BOTH the Vulkan deviceLUID (forced into
// VkPhysicalDeviceIDProperties) and the nvapi OS-AdapterId, so NGX's adapter correlation matches.
// DERIVED from the fixed deviceUUID (set in s_vkGPDP2) — deterministic + stable across process
// runs + collision-free, in case NGX uses the LUID as a cache key for its model/config files.
static u8 g_luid[8]={0x70,0x50,0xB1,0xAC,0x4E,0x11,0x00,0x00};  // fallback until GPDP2 fills it
// NV_LOGICAL_GPU_DATA: version@0, pOSAdapterId@8, physicalGpuCount@16, physicalGpuHandles@24.
MSABI static int s_GPU_GetLogicalGpuInfo(void* lgpu,u8* d){ (void)lgpu; logs("[LGI entered]\n"); if(!d) return 0;
    void* osid=*(void**)(d+8); if(osid) memcpy(osid,g_luid,8);
    *(u32*)(d+16)=1; *(void**)(d+24)=FAKE_GPU; logs("[LGI filled]\n"); return 0; }
MSABI static void* s_nvapi_QueryInterface(u32 id){
    void* f=0; const char* nm="?";
    switch(id){
        case 0x0150E828: f=(void*)s_NvAPI_Initialize; nm="Initialize"; break;
        case 0xD22BDD7E: f=(void*)s_NvAPI_Unload; nm="Unload"; break;
        case 0x0694D52E: f=(void*)s_DRS_CreateSession; nm="DRS_CreateSession"; break;
        case 0x375DBD6B: f=(void*)s_DRS_LoadSettings; nm="DRS_LoadSettings"; break;
        case 0xDA8466A0: f=(void*)s_DRS_GetBaseProfile; nm="DRS_GetBaseProfile"; break;
        case 0xDAD9CFF8: f=(void*)s_DRS_DestroySession; nm="DRS_DestroySession"; break;
        case 0xEEE566B2: f=(void*)s_DRS_FindApplicationByName; nm="DRS_FindApplicationByName"; break;
        case 0x61CD6FD6: f=(void*)s_DRS_GetProfileInfo; nm="DRS_GetProfileInfo"; break;
        case 0x73BF8338: f=(void*)s_DRS_GetSetting; nm="DRS_GetSetting"; break;
        case 0x2926AAAD: f=(void*)s_SYS_GetDriverAndBranchVersion; nm="SYS_GetDriverAndBranchVersion"; break;
        case 0xE5AC921F: f=(void*)s_EnumPhysicalGPUs; nm="EnumPhysicalGPUs"; break;
        case 0xD8265D24: f=(void*)s_GPU_GetArchInfo; nm="GPU_GetArchInfo(Blackwell)"; break;
        case 0xADD604D1: f=(void*)s_GetLogicalGPU; nm="GetLogicalGPUFromPhysicalGPU"; break;
        case 0x842B066E: f=(void*)s_GPU_GetLogicalGpuInfo; nm="GPU_GetLogicalGpuInfo"; break;
        case 0x2DDFB66E: f=(void*)s_GPU_GetPCIIdentifiers; nm="GPU_GetPCIIdentifiers"; break;
        case 0xCEEE8E9F: f=(void*)s_GPU_GetFullName; nm="GPU_GetFullName"; break;
        case 0xC33BAEB1: f=(void*)s_GPU_GetGPUType; nm="GPU_GetGPUType"; break;
        case 0x1BB18724: f=(void*)s_GPU_GetBusType; nm="GPU_GetBusType"; break;
        default: nm="(unimpl->null)"; break;
    }
    char b[64]; snprintf(b,sizeof b,"[nvapi_QI 0x%08X %s]\n",id,nm); logs(b); return f; }

// forward decls
static void* resolve(const char* dll,const char* fn);
static void* make_trap(const char* name);
static Module* load_module(const char* name);
static void* module_export(Module* m,const char* fn);

// dynamic loader entry points (real)
MSABI static void* s_GetModuleHandleW(void* n){ (void)n; return g_mod[0].base; }
MSABI static void* s_GetProcAddress(void* m,const char* n){ if(!n)return 0;
    for(int i=0;i<g_nmod;i++) if(g_mod[i].base==(u8*)m){ void* e=module_export(&g_mod[i],n); if(e)return e; break; }
    // Vulkan loader: the host's GPU-arch path LoadLibrary's vulkan-1.dll + resolves entry
    // points. gipa/gdpa must be our ms_abi wrappers (they ms2sysv-wrap what they return);
    // other vk* funcs bridge straight to native libvulkan via ms2sysv.
    if(!strcmp(n,"vkGetInstanceProcAddr")){ logn("[vk->native] ",n); return (void*)my_gipa; }
    if(!strcmp(n,"vkGetDeviceProcAddr")){ logn("[vk->native] ",n); return (void*)my_gdpa; }
    if(!strncmp(n,"vk",2)&&g_hvk){ void* f=dlsym(g_hvk,n); if(f){ logn("[vk->native] ",n); return make_ms2sysv(f); } }
    // NVML is public + native: bridge nvml.dll's functions to libnvidia-ml.so.1 via ms2sysv.
    if(!strncmp(n,"nvml",4)&&g_hnvml){ void* f=dlsym(g_hnvml,n); if(f){ logn("[nvml->native] ",n); return make_ms2sysv(f); } }
    // fall back to our CRT stub table by name, else a logging trap
    void* r=resolve("dyn",n); return r; }
static void wide_basename(const void* w,char* out){ const u16* p=w; int n=0; char tmp[260];
    for(;p[n]&&n<259;n++) tmp[n]=(char)(p[n]&0xFF); tmp[n]=0; char* s=strrchr(tmp,'\\'); s=s?s+1:tmp; char* s2=strrchr(s,'/'); s=s2?s2+1:s; strcpy(out,s); }
MSABI static void* s_LoadLibraryExW(void* n,void* h,u32 f){ (void)h;(void)f; char base[260]; wide_basename(n,base);
    logn("[LoadLibrary] ",base); Module* m=load_module(base); return m?m->base:(void*)0x140000000ULL; }
MSABI static void* s_LoadLibraryA(const char* n){ const char* s=strrchr(n,'\\'); s=s?s+1:n; logn("[LoadLibraryA] ",s);
    Module* m=load_module(s); return m?m->base:(void*)0x140000000ULL; }
MSABI static int   s_GetModuleHandleExW(u32 f,void* n,void** out){ (void)f;(void)n; if(out)*out=g_mod[0].base; return 1; }

struct { const char* name; void* fn; } g_stubs[]={
 {"GetProcessHeap",s_GetProcessHeap},{"HeapAlloc",s_HeapAlloc},{"HeapReAlloc",s_HeapReAlloc},{"HeapFree",s_HeapFree},
 {"HeapSize",s_HeapSize},{"HeapCreate",s_HeapCreate},{"HeapDestroy",s_HeapDestroy},{"LocalAlloc",s_LocalAlloc},{"LocalFree",s_LocalFree},
 {"TlsAlloc",s_TlsAlloc},{"TlsGetValue",s_TlsGetValue},{"TlsSetValue",s_TlsSetValue},{"TlsFree",s_TlsFree},
 {"FlsAlloc",s_TlsAlloc},{"FlsGetValue",s_TlsGetValue},{"FlsSetValue",s_TlsSetValue},{"FlsFree",s_TlsFree},
 {"GetCurrentThreadId",s_GetCurrentThreadId},{"GetCurrentProcessId",s_GetCurrentProcessId},
 {"GetCurrentProcess",s_GetCurrentProcess},{"GetCurrentThread",s_GetCurrentThread},
 {"GetSystemTimeAsFileTime",s_GetSystemTimeAsFileTime},{"GetSystemTimePreciseAsFileTime",s_GetSystemTimeAsFileTime},
 {"QueryPerformanceCounter",s_QueryPerformanceCounter},{"QueryPerformanceFrequency",s_QueryPerformanceFrequency},
 {"EncodePointer",s_EncodePointer},{"DecodePointer",s_DecodePointer},{"IsProcessorFeaturePresent",s_IsProcessorFeaturePresent},
 {"IsDebuggerPresent",s_IsDebuggerPresent},{"InitializeSListHead",s_InitializeSListHead},{"InterlockedFlushSList",s_ret0p},
 {"GetLastError",s_GetLastError},{"SetLastError",s_SetLastError},
 {"GetModuleHandleW",s_GetModuleHandleW},{"GetModuleHandleA",s_GetModuleHandleW},{"GetProcAddress",s_GetProcAddress},
 {"GetModuleHandleExW",s_GetModuleHandleExW},{"GetModuleHandleExA",s_GetModuleHandleExW},
 {"LoadLibraryW",s_LoadLibraryExW},{"LoadLibraryExW",s_LoadLibraryExW},{"LoadLibraryA",s_LoadLibraryA},{"LoadLibraryExA",s_LoadLibraryA},
 {"GetStartupInfoW",s_GetStartupInfoW},{"GetCommandLineW",s_GetCommandLineW},{"GetCommandLineA",s_GetCommandLineW},
 {"OutputDebugStringA",s_OutputDebugStringA},
 {"GetEnvironmentStringsW",s_GetEnvironmentStringsW},{"FreeEnvironmentStringsW",s_FreeEnvironmentStringsW},{"GetEnvironmentVariableA",s_GetEnvironmentVariableA},
 {"WideCharToMultiByte",s_WideCharToMultiByte},{"MultiByteToWideChar",s_MultiByteToWideChar},
 {"InitOnceExecuteOnce",s_InitOnceExecuteOnce},
 {"InitializeCriticalSection",s_noop},{"InitializeCriticalSectionEx",s_ret1},{"InitializeCriticalSectionAndSpinCount",s_ret1},
 {"EnterCriticalSection",s_noop},{"LeaveCriticalSection",s_noop},{"DeleteCriticalSection",s_noop},{"TryEnterCriticalSection",s_ret1},
 {"InitializeSRWLock",s_noop},{"AcquireSRWLockExclusive",s_noop},{"ReleaseSRWLockExclusive",s_noop},{"TryAcquireSRWLockExclusive",s_ret1},
 {"AcquireSRWLockShared",s_noop},{"ReleaseSRWLockShared",s_noop},
 {"InitializeConditionVariable",s_noop},{"WakeConditionVariable",s_noop},{"WakeAllConditionVariable",s_noop},
 {"SleepConditionVariableCS",s_ret1},{"SleepConditionVariableSRW",s_ret1},
 {"CreateEventW",s_CreateHandle},{"CreateEventExW",s_CreateHandle},{"CreateEventA",s_CreateHandle},
 {"CreateSemaphoreW",s_CreateHandle},{"CreateSemaphoreExW",s_CreateHandle},{"CreateMutexW",s_CreateHandle},{"CreateMutexExW",s_CreateHandle},
 {"CloseHandle",s_CloseHandle},{"WaitForSingleObject",s_WaitForSingleObject},{"WaitForSingleObjectEx",s_WaitForSingleObject},
 {"SetEvent",s_ret1},{"ResetEvent",s_ret1},{"GetTickCount64",s_GetTickCount64},
 {"SetUnhandledExceptionFilter",s_ret0p},{"UnhandledExceptionFilter",s_ret1},{"RtlLookupFunctionEntry",s_ret0p},{"RtlPcToFileHeader",s_ret0p},
 {"GetModuleFileNameW",s_GetModuleFileNameW},{"GetModuleFileNameA",s_GetModuleFileNameA},
 {"GetSystemDirectoryW",s_GetSystemDirectoryW},{"GetWindowsDirectoryW",s_GetWindowsDirectoryW},
 {"VerifyVersionInfoW",s_VerifyVersionInfoW},{"VerSetConditionMask",s_VerSetConditionMask},{"GetFullPathNameW",s_GetFullPathNameW},
 {"nvapi_QueryInterface",s_nvapi_QueryInterface},
 {"CreateFileW",s_CreateFileW},{"GetFileAttributesW",s_GetFileAttributesW},{"GetFileAttributesExW",s_GetFileAttributesExW},{"FindFirstFileExW",s_FindFirstFileExW},
 {"SHGetKnownFolderPath",s_SHGetKnownFolderPath},{"CoTaskMemFree",s_CoTaskMemFree},
 {"D3DKMTEnumAdapters2",s_D3DKMTEnumAdapters2},{"PathFileExistsW",s_PathFileExistsW},
 {"D3DKMTQueryAdapterInfo",s_D3DKMTQueryAdapterInfo},{"D3DKMTCloseAdapter",s_D3DKMTCloseAdapter},{"D3DKMTOpenAdapterFromLuid",s_D3DKMTOpenAdapterFromLuid},
 {0,0}};
static void* g_stubs_lookup(const char* fn){ for(int i=0;g_stubs[i].name;i++) if(!strcmp(g_stubs[i].name,fn)) return g_stubs[i].fn; return 0; }

MSABI static u64 trap_log(const char* name){ logn("[STUB] ",name); return 0; }
static void* make_trap(const char* name){ u8* p=g_code+g_codeoff,*st=p;
    *p++=0x48;*p++=0x83;*p++=0xEC;*p++=0x28; *p++=0x48;*p++=0xB9;memcpy(p,&name,8);p+=8;
    void* lg=(void*)trap_log; *p++=0x48;*p++=0xB8;memcpy(p,&lg,8);p+=8; *p++=0xFF;*p++=0xD0;
    *p++=0x48;*p++=0x83;*p++=0xC4;*p++=0x28; *p++=0xC3; g_codeoff+=(size_t)(p-st); return st; }
static void* g_hvk,*g_hcu;   // native libvulkan / libcuda handles
static void* make_ms2sysv(void* target){ if(!target) return 0; u8* p=g_code+g_codeoff,*st=p;
    *p++=0x49;*p++=0xBA;memcpy(p,&target,8);p+=8;             // movabs r10, target
    void* c=(void*)ms2sysv_common; *p++=0x48;*p++=0xB8;memcpy(p,&c,8);p+=8;  // movabs rax, common
    *p++=0xFF;*p++=0xE0;                                      // jmp rax
    g_codeoff+=(size_t)(p-st); return st; }
static void* resolve(const char* dll,const char* fn){
    // Vulkan/CUDA imports -> native driver via an ms_abi->SysV thunk (the PE calls MS-x64).
    if(!strcasecmp(dll,"vulkan-1.dll")){ void* n=g_hvk?dlsym(g_hvk,fn):0; if(n) return make_ms2sysv(n); }
    if(!strcasecmp(dll,"nvcuda.dll")){   void* n=g_hcu?dlsym(g_hcu,fn):0; if(n) return make_ms2sysv(n); }
    void* s=g_stubs_lookup(fn); if(s)return s;
    char* nm=malloc(strlen(dll)+strlen(fn)+2); sprintf(nm,"%s:%s",dll,fn); return make_trap(nm); }

static void* module_export(Module* m,const char* fn){ if(!m->exp_rva) return 0; const u8* e=m->base+m->exp_rva;
    u32 nnames=rd32(e+0x18),funcs=rd32(e+0x1C),names=rd32(e+0x20),ords=rd32(e+0x24);
    const u8* na=m->base+names; const u8* oa=m->base+ords; const u8* fa=m->base+funcs;
    for(u32 i=0;i<nnames;i++){ const char* nm=(const char*)(m->base+rd32(na+i*4)); if(!strcmp(nm,fn)){ u16 o=rd16(oa+i*2); return m->base+rd32(fa+o*4);} }
    return 0; }

// map + relocate + wire imports + exec-protect + run DllMain; register + parse exports.
static Module* load_module(const char* name){
    for(int i=0;i<g_nmod;i++) if(!strcasecmp(g_mod[i].name,name)) return &g_mod[i];   // already loaded
    char path[320]; snprintf(path,sizeof path,WINE"%s",name);
    int fd=open(path,O_RDONLY); if(fd<0){ logn("[load: not found] ",name); return 0; }
    struct stat stt; fstat(fd,&stt); u8* file=mmap(NULL,stt.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(file==MAP_FAILED||rd16(file)!=0x5A4D) return 0;
    u32 e=rd32(file+0x3C); const u8* nt=file+e; if(rd32(nt)!=0x4550) return 0;
    const u8* fh=nt+4; const u8* oh=fh+20; u16 nsec=rd16(fh+2),optsz=rd16(fh+16);
    u64 imgbase=rd64(oh+24); u32 sizeimg=rd32(oh+56),sizehdr=rd32(oh+60),ndir=rd32(oh+108); const u8* dir=oh+112;
    u32 expr=ndir>0?rd32(dir+0):0,exps=ndir>0?rd32(dir+4):0,imp=ndir>1?rd32(dir+8):0,rel=ndir>5?rd32(dir+40):0,relsz=ndir>5?rd32(dir+44):0;
    u32 tls=ndir>9?rd32(dir+72):0, entry=rd32(oh+16);
    u8* base=mmap(NULL,sizeimg,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    memcpy(base,file,sizehdr); const u8* sec=oh+optsz;
    for(u16 i=0;i<nsec;i++){const u8* s=sec+i*40;u32 vs=rd32(s+8),va=rd32(s+12),rs=rd32(s+16),rp=rd32(s+20);u32 n=rs<vs?rs:vs;if(va+n<=sizeimg&&rp+n<=stt.st_size)memcpy(base+va,file+rp,n);}
    int64_t delta=(int64_t)((u64)base-imgbase);
    if(rel&&delta){const u8* p=base+rel,*end=p+relsz;while(p+8<=end){u32 pg=rd32(p),bl=rd32(p+4);if(bl<8)break;for(u32 i=0;i<(bl-8)/2;i++){u16 en=rd16(p+8+i*2);if((en>>12)==10){u8* t=base+pg+(en&0xFFF);u64 v=rd64(t)+delta;memcpy(t,&v,8);}}p+=bl;}}
    if(imp){const u8* d=base+imp;for(;;d+=20){u32 oft=rd32(d),nm=rd32(d+12),ft=rd32(d+16);if(!nm&&!ft&&!oft)break;const char* dll=(const char*)(base+nm);const u8* t=base+(oft?oft:ft);u8* iat=base+ft;
        for(;;t+=8,iat+=8){u64 v=rd64(t);if(!v)break;if(v&0x8000000000000000ULL)continue;const char* fn=(const char*)(base+(u32)v+2);void* r=resolve(dll,fn);memcpy(iat,&r,8);}}}
    for(u16 i=0;i<nsec;i++){const u8* s=sec+i*40;u32 va=rd32(s+12),vs=rd32(s+8),ch=rd32(s+36);u32 len=(vs+0xFFF)&~0xFFFu;int prot=PROT_READ;if(ch&0x80000000)prot|=PROT_WRITE;if(ch&0x20000000)prot|=PROT_EXEC;if(va+len<=sizeimg)mprotect(base+va,len,prot);}
    Module* m=&g_mod[g_nmod++]; strncpy(m->name,name,63); m->base=base; m->imgbase=imgbase; m->exp_rva=expr; m->exp_sz=exps; m->loaded=1;
    // PE-TLS block for this module -> hang off the shared gs:[0x58] TLS array (set in main)
    if(tls){const u8* td=base+tls;u64 start=rd64(td+0),endr=rd64(td+8),idxaddr=rd64(td+16);u32 zf=rd32(td+32);
        u64 raw=endr-start;u8* blk=calloc(1,raw+zf+64);if(raw)memcpy(blk,(void*)start,raw);
        u32 slot=(u32)(m-g_mod)+1; if(slot<512){*(u32*)idxaddr=slot; g_tlsslots[slot]=blk;} }
    // run DllMain
    typedef int MSABI(*dm_t)(void*,u32,void*); dm_t dm=(dm_t)(base+entry);
    logn("[loading] ",name); int r=dm((void*)base,1,0);
    { char b[64]; snprintf(b,sizeof b,"[%s DllMain -> %d, exports@rva=0x%x]\n",name,r,expr); logs(b);}
    return m;
}

// --- live Vulkan device (native SysV; created before gs is switched) ---
static VkInstance g_inst; static VkPhysicalDevice g_pd; static VkDevice g_dev; static uint32_t g_qfam; static VkQueue g_queue;
static int try_device(const char** de,uint32_t nde){ float pr=1;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=g_qfam,.queueCount=1,.pQueuePriorities=&pr};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=nde,.ppEnabledExtensionNames=de};
    return vkCreateDevice(g_pd,&dci,0,&g_dev)==VK_SUCCESS?0:1; }
static int setup_vulkan(void){
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
    const char* ie[]={"VK_KHR_get_physical_device_properties2","VK_KHR_external_memory_capabilities","VK_KHR_external_semaphore_capabilities"};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app,.enabledExtensionCount=3,.ppEnabledExtensionNames=ie};
    if(vkCreateInstance(&ici,0,&g_inst)!=VK_SUCCESS) return 1;
    uint32_t n=0; vkEnumeratePhysicalDevices(g_inst,&n,0); VkPhysicalDevice pds[8]; if(n>8)n=8; vkEnumeratePhysicalDevices(g_inst,&n,pds);
    for(uint32_t i=0;i<n;i++){VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(pds[i],&p);if(p.vendorID==0x10DE){g_pd=pds[i];break;}}
    if(!g_pd) return 2;
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&qn,0); VkQueueFamilyProperties q[16]; if(qn>16)qn=16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&qn,q);
    for(uint32_t i=0;i<qn;i++) if(q[i].queueFlags&VK_QUEUE_GRAPHICS_BIT){g_qfam=i;break;}
    const char* de[]={"VK_KHR_external_memory","VK_KHR_external_memory_fd","VK_KHR_external_semaphore","VK_KHR_external_semaphore_fd","VK_KHR_push_descriptor"};
    if(try_device(de,5) && try_device(de,3) && try_device(0,0)) return 3;   // degrade until a device is created
    vkGetDeviceQueue(g_dev,g_qfam,0,&g_queue); return 0; }
// ms_abi gipa/gdpa: the host calls these MS-x64; return ms2sysv-wrapped native entry points.
// Interpose vkGetPhysicalDeviceProperties2 (the ONE fn NGX's arch path resolves): call
// native, then log the device id NGX sees + the pNext structs it asked to be filled.
MSABI static void s_vkGPDP2(void* pd,void* pprops){
    vkGetPhysicalDeviceProperties2((VkPhysicalDevice)pd,(VkPhysicalDeviceProperties2*)pprops);
    VkPhysicalDeviceProperties2* p2=(VkPhysicalDeviceProperties2*)pprops;
    char b[160]; snprintf(b,sizeof b,"[GPDP2] vendor=0x%04X device=0x%04X name=%s\n",
        p2->properties.vendorID,p2->properties.deviceID,p2->properties.deviceName); logs(b);
    for(VkBaseOutStructure* s=(VkBaseOutStructure*)p2->pNext;s;s=s->pNext){
        if(s->sType==VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES){
            VkPhysicalDeviceIDProperties* idp=(VkPhysicalDeviceIDProperties*)s;
            char c[96]; snprintf(c,sizeof c,"   ID_PROPS: LUIDValid=%u LUID=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                idp->deviceLUIDValid,idp->deviceLUID[0],idp->deviceLUID[1],idp->deviceLUID[2],idp->deviceLUID[3],
                idp->deviceLUID[4],idp->deviceLUID[5],idp->deviceLUID[6],idp->deviceLUID[7]); logs(c);
            // The native NVIDIA Linux driver reports deviceLUIDValid=0. Synthesize a valid LUID
            // DERIVED from the fixed deviceUUID (first 8 bytes) — stable + collision-free — and use
            // the SAME bytes as the nvapi OS-AdapterId (g_luid) so NGX's adapter correlation matches.
            if(!idp->deviceLUIDValid){ memcpy(g_luid,idp->deviceUUID,8); idp->deviceLUIDValid=1; memcpy(idp->deviceLUID,g_luid,8); idp->deviceNodeMask=1; }
        } else { char c[48]; snprintf(c,sizeof c,"   pNext sType=%u\n",(unsigned)s->sType); logs(c);} } }
MSABI static void* my_gipa(void* inst,const char* n){ if(n)logn("[gipa] ",n);
    if(n&&!strcmp(n,"vkGetPhysicalDeviceProperties2")) return (void*)s_vkGPDP2;
    void* f=(void*)vkGetInstanceProcAddr((VkInstance)inst,n); return f?make_ms2sysv(f):0; }
MSABI static void* my_gdpa(void* dev,const char* n){ if(n)logn("[gdpa] ",n); PFN_vkGetDeviceProcAddr g=(PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(g_inst,"vkGetDeviceProcAddr"); void* f=g?(void*)g((VkDevice)dev,n):0; return f?make_ms2sysv(f):0; }

static void hex64(char* o,u64 a){ for(int i=0;i<16;i++){int d=(a>>((15-i)*4))&0xF;o[i]=d<10?'0'+d:'a'+d-10;} o[16]=0; }
static void segv(int s,siginfo_t* si,void* uc){ (void)s;
    u64 addr=(u64)si->si_addr; u64 rip=0;
    if(uc){ ucontext_t* c=(ucontext_t*)uc; rip=(u64)c->uc_mcontext.gregs[REG_RIP]; }
    char b[20]; char out[256]="[SIGSEGV addr=0x"; hex64(b,addr); strcat(out,b); strcat(out," rip=0x"); hex64(b,rip); strcat(out,b);
    // which loaded module is rip in?
    for(int i=0;i<g_nmod;i++){ u64 lo=(u64)g_mod[i].base; if(rip>=lo&&rip<lo+0x2000000){ strcat(out," in "); strcat(out,g_mod[i].name); strcat(out,"+0x"); hex64(b,rip-lo); strcat(out,b); break; } }
    if(rip>=(u64)g_code&&rip<(u64)g_code+(1<<20)) strcat(out," in <thunk/trap>");
    strcat(out,"]\n"); (void)write(2,out,strlen(out)); _exit(42); }

int main(void){
    printf("== Path B / S5(a): load Windows NGX host natively\n");
    g_code=mmap(NULL,1<<20,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    // module[0] must exist for GetModuleHandle(self); reserve a dummy base
    fflush(stdout);
    pid_t pid=fork();
    if(pid==0){
        struct sigaction sa;memset(&sa,0,sizeof sa);sa.sa_sigaction=segv;sa.sa_flags=SA_SIGINFO;
        sigaction(SIGSEGV,&sa,0);sigaction(SIGILL,&sa,0);sigaction(SIGBUS,&sa,0);sigaction(SIGFPE,&sa,0);
        // native drivers for the ms_abi->SysV import thunks
        g_hvk=dlopen("libvulkan.so.1",RTLD_NOW|RTLD_GLOBAL); g_hcu=dlopen("libcuda.so.1",RTLD_NOW|RTLD_GLOBAL);
        g_hnvml=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_GLOBAL);
        int vr=setup_vulkan();
        { char b[96]; snprintf(b,sizeof b,"[vulkan setup rc=%d inst=%p pd=%p dev=%p]\n",vr,(void*)g_inst,(void*)g_pd,(void*)g_dev); logs(b);}
        if(vr){ logs("[no vulkan device -> cannot call NGX Init]\n"); _exit(5); }
        // Windows TEB on gs (see S2). Created AFTER the native Vulkan device.
        memset(g_teb,0,sizeof g_teb);memset(g_peb,0,sizeof g_peb);memset(g_tlsslots,0,sizeof g_tlsslots);
        *(void**)(g_teb+0x30)=g_teb;*(void**)(g_teb+0x58)=g_tlsslots;*(void**)(g_teb+0x60)=g_peb;
        u8* sp;__asm__("mov %%rsp,%0":"=r"(sp));*(void**)(g_teb+0x08)=sp+0x100000;*(void**)(g_teb+0x10)=sp-0x400000;
        syscall(SYS_arch_prctl,0x1001,(unsigned long)g_teb);
        // load the NGX core host; it LoadLibrary's its siblings (logged)
        Module* h=load_module("_nvngx.dll");
        if(!h){ logs("host load failed\n"); _exit(3); }
        static u16 wpath[80]; const char* dp=WINE; int i=0; for(;dp[i]&&i<79;i++) wpath[i]=(u8)dp[i]; wpath[i]=0;  // UTF-16 (2-byte) path

        // LIGHTER query first: NVSDK_NGX_VULKAN_GetFeatureRequirements(FrameGeneration) — the
        // same call that returned GREEN under Proton. It takes only instance+pd, NOT the full
        // Init/adapter-correlation path (dxvk-nvapi's GetLogicalGpuInfo/LUID is never touched
        // here). Tests whether our native nvapi bridge (arch=Blackwell) is enough for the
        // driver to report FG available NATIVELY. FeatureRequirement = {FeatureSupported@0(u32),
        // MinHWArch@4(u32), MinOS[255]}; FeatureDiscoveryInfo = {SDKVer,FeatureID,Ident(32),path,info}.
        typedef int MSABI(*gfr_t)(void*,void*,const void*,void*);
        gfr_t GFR=(gfr_t)module_export(h,"NVSDK_NGX_VULKAN_GetFeatureRequirements");
        if(GFR){
            struct { u32 sdkVer,featureId,idType,pad; u64 appId,u1,u2; const void* dataPath; const void* featInfo; } fdi;
            memset(&fdi,0,sizeof fdi); fdi.sdkVer=0x15; fdi.featureId=11; fdi.idType=0; fdi.appId=0x1337ULL; fdi.dataPath=wpath;
            struct { u32 fsupp,minhw; char minos[256]; } frq; memset(&frq,0,sizeof frq);
            logs("\n[calling native NVSDK_NGX_VULKAN_GetFeatureRequirements(FrameGeneration=11) ...]\n");
            int rr=GFR((void*)g_inst,(void*)g_pd,&fdi,&frq);
            char b[160]; snprintf(b,sizeof b,"[native FG requirements: result=0x%08X FeatureSupported=0x%X (0=SUPPORTED) MinHWArch=0x%X MinOS=%.16s]\n",
                (unsigned)rr,frq.fsupp,frq.minhw,frq.minos); logs(b);
        } else logs("GetFeatureRequirements export missing\n");

        // Full Init (the heavier path that still crashes in adapter correlation — kept for
        // the CreateFeature work). Skipped for the requirements test.
        typedef int MSABI(*init_t)(const char*,int,const char*,const u16*,void*,void*,void*,void*,void*,const void*,unsigned);
        init_t Init=(init_t)module_export(h,"NVSDK_NGX_VULKAN_Init_ProjectID");
        if(Init && getenv("RUN_INIT")){
            logs("\n[calling NVSDK_NGX_VULKAN_Init_ProjectID ...]\n");
            int r=Init("a0b1c2d3-1234-5678-9abc-def012345678",0,"1.0",wpath,
                       (void*)g_inst,(void*)g_pd,(void*)g_dev,(void*)my_gipa,(void*)my_gdpa,0,0x15);
            char b[80]; snprintf(b,sizeof b,"[NGX Init returned 0x%08X]\n",(unsigned)r); logs(b);
        }
        _exit(0);
    }
    int stx; waitpid(pid,&stx,0);
    if(WIFEXITED(stx)) printf("\n== child exit %d\n",WEXITSTATUS(stx));
    else if(WIFSIGNALED(stx)) printf("\n== child signal %d\n",WTERMSIG(stx));
    printf("== S5(a) verdict: see host DllMain result + dynamic LoadLibrary trace above.\n");
    return 0;
}
