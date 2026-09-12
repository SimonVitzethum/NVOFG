// Oracle DATA probe: under Proton, call the SAME nvapi functions dxvk-nvapi implements that the
// NGX FG evaluator reads (GPU_GetArchInfo, GPU_GetLogicalGpuInfo, GetPCIIdentifiers, ...) and dump
// the exact bytes dxvk-nvapi returns. Diff against our native stub values (0x1B0/impl=5/rev=0xA1,
// g_luid, ...) to find the one DATA field that diverges and gates FeatureRequirements(FG).
//
// Build: x86_64-w64-mingw32-gcc -O2 -o nvapi_dump.exe nvapi_dump.c
// Run  : under GE-Proton (PROTON_ENABLE_NVAPI=1), writes nvapi_dump_result.txt
#include <windows.h>
#include <stdio.h>
typedef void* (*QI_t)(unsigned id);
typedef int (*Init_t)(void);
typedef int (*Enum_t)(void** handles, unsigned* count);
typedef int (*Arch_t)(void* gpu, unsigned* archinfo);
typedef int (*GetLog_t)(void* phys, void** logical);
typedef int (*LogInfo_t)(void* logical, unsigned char* data);
typedef int (*PCIId_t)(void* gpu, unsigned* dev, unsigned* sub, unsigned* rev, unsigned* ext);
typedef int (*BusId_t)(void* gpu, unsigned* busId);
static FILE* g;
static void L(const char* f,...){ va_list a; va_start(a,f); vprintf(f,a); va_end(a); if(g){ va_start(a,f); vfprintf(g,f,a); va_end(a); fflush(g);} }

int main(void){
    g=fopen("nvapi_dump_result.txt","w");
    HMODULE h=LoadLibraryA("nvapi64.dll");
    if(!h){ L("no nvapi64.dll\n"); return 1; }
    QI_t QI=(QI_t)(void*)GetProcAddress(h,"nvapi_QueryInterface");
    if(!QI){ L("no QI\n"); return 1; }
    Init_t Init=(Init_t)QI(0x0150E828);
    Enum_t Enum=(Enum_t)QI(0xE5AC921F);
    Arch_t Arch=(Arch_t)QI(0xD8265D24);
    GetLog_t GetLog=(GetLog_t)QI(0xADD604D1);
    LogInfo_t LogInfo=(LogInfo_t)QI(0x842B066E);
    PCIId_t PCIId=(PCIId_t)QI(0x2DDFB66E);
    BusId_t BusId=(BusId_t)QI(0x1BE0B8E5);
    L("QI: Init=%p Enum=%p Arch=%p GetLog=%p LogInfo=%p PCIId=%p\n",(void*)Init,(void*)Enum,(void*)Arch,(void*)GetLog,(void*)LogInfo,(void*)PCIId);
    if(Init) L("Init -> %d\n", Init());
    void* gpus[64]; unsigned n=0;
    int er = Enum? Enum(gpus,&n) : -999;
    L("EnumPhysicalGPUs -> %d count=%u gpu0=%p\n", er, n, n?gpus[0]:0);
    if(!n){ L("no gpus\n"); return 1; }
    void* gpu=gpus[0];
    // GPU_GetArchInfo — try version 2 then 1
    for(int ver=2; ver>=1; --ver){
        unsigned ai[16]; memset(ai,0,sizeof ai);
        unsigned sz = (ver==2)? 7*4 : 4*4;
        ai[0] = sz | (ver<<16);              // NV_GPU_ARCH_INFO_VER
        int r = Arch? Arch(gpu, ai) : -999;
        L("GPU_GetArchInfo(v%d) -> %d : arch=0x%X impl=0x%X rev=0x%X  [+extra: 0x%X 0x%X 0x%X]\n",
          ver, r, ai[1], ai[2], ai[3], ai[4], ai[5], ai[6]);
    }
    if(PCIId){ unsigned dev=0,sub=0,rev=0,ext=0; int r=PCIId(gpu,&dev,&sub,&rev,&ext);
        L("GPU_GetPCIIdentifiers -> %d : dev=0x%08X sub=0x%08X rev=0x%08X ext=0x%08X\n", r,dev,sub,rev,ext); }
    // logical GPU + OS adapter id
    void* logical=0; int rl = GetLog? GetLog(gpu,&logical) : -999;
    L("GetLogicalGPUFromPhysicalGPU -> %d logical=%p\n", rl, logical);
    if(logical && LogInfo){
        // EXACT version measured at our native stub: 0x00010238 (size=568, ver=1).
        unsigned char buf[1024]; memset(buf,0,sizeof buf);
        unsigned char osid[16]; memset(osid,0xEE,sizeof osid);   // sentinel: did dxvk overwrite it?
        *(unsigned*)(buf+0) = 0x00010238;                        // 568 | (1<<16)
        *(void**)(buf+8) = osid;                                 // pOSAdapterId
        int r = LogInfo(logical, buf);
        L("GPU_GetLogicalGpuInfo(0x10238) -> %d\n", r);
        L("  OSAdapterId = %02x%02x%02x%02x%02x%02x%02x%02x\n",
          osid[0],osid[1],osid[2],osid[3],osid[4],osid[5],osid[6],osid[7]);
        // dump the whole 568-byte struct in 8-byte rows so we can diff every field vs our stub
        for(int off=0; off<568; off+=16){
            L("  +%03x: %02x%02x%02x%02x %02x%02x%02x%02x  %02x%02x%02x%02x %02x%02x%02x%02x\n", off,
              buf[off+0],buf[off+1],buf[off+2],buf[off+3],buf[off+4],buf[off+5],buf[off+6],buf[off+7],
              buf[off+8],buf[off+9],buf[off+10],buf[off+11],buf[off+12],buf[off+13],buf[off+14],buf[off+15]);
        }
    }
    L("DONE\n");
    return 0;
}
