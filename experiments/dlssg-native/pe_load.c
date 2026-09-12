// Path B — S2: attempt to actually RUN nvngx_dlssg.dll's DllMain in a native Linux
// process. Extends the S1 probe (map + relocate + resolve) with:
//   * executable section protections,
//   * the Microsoft-x64 ABI boundary: a PE calls its imports MS-x64 (rcx,rdx,r8,r9 +
//     32B shadow), NOT SysV — so every stub is __attribute__((ms_abi)), and unknown
//     imports get a generated ms_abi trampoline that logs its own name and returns 0,
//   * a minimal MSVC-CRT shim (heap/TLS/time/sync/module) — enough to try to reach the
//     C++ static initialisers + DllMain,
//   * running TLS callbacks + DllMain(DLL_PROCESS_ATTACH) in a forked child under a
//     SIGSEGV handler, logging the import-call trace and the first blocker.
//
// This is an S2 *observation* run: the goal is to learn how far CRT init gets and what
// it calls, not (yet) a working feature. Vulkan/CUDA imports are trap-logged for now
// (they're only called later, during CreateFeature/Evaluate — S5/S6).
//
// Build: gcc -O2 -o pe_load pe_load.c -ldl   Run: ./pe_load [path]
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
#include <time.h>
#include <unistd.h>

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
#define MSABI __attribute__((ms_abi))
static u32 rd32(const u8* p){u32 v;memcpy(&v,p,4);return v;}
static u16 rd16(const u8* p){u16 v;memcpy(&v,p,2);return v;}
static u64 rd64(const u8* p){u64 v;memcpy(&v,p,8);return v;}

// ---- tiny async-safe logging (child writes directly to fd 2) ----
static void logs(const char* s){ (void)write(2, s, strlen(s)); }
static void logn(const char* pfx, const char* s){ logs(pfx); logs(s); logs("\n"); }

// ---- minimal MSVC-CRT shim state ----
static __thread void* g_tls[2048];
static __thread int   g_lasterr;
static int g_tlsnext = 1;
static u8  g_heap[1];              // fake heap handle target

MSABI static void* s_GetProcessHeap(void){ return g_heap; }
MSABI static void* s_HeapAlloc(void* h,u32 fl,u64 sz){ (void)h; return (fl&8)?calloc(1,sz):malloc(sz); }
MSABI static void* s_HeapReAlloc(void* h,u32 fl,void* p,u64 sz){ (void)h;(void)fl; return realloc(p,sz); }
MSABI static int   s_HeapFree(void* h,u32 fl,void* p){ (void)h;(void)fl; free(p); return 1; }
MSABI static u64   s_HeapSize(void* h,u32 fl,void* p){ (void)h;(void)fl;(void)p; return 0; }
MSABI static void* s_HeapCreate(u32 a,u64 b,u64 c){ (void)a;(void)b;(void)c; return g_heap; }
MSABI static int   s_HeapDestroy(void* h){ (void)h; return 1; }
MSABI static void* s_LocalAlloc(u32 fl,u64 sz){ return (fl&0x40)?calloc(1,sz):malloc(sz); }
MSABI static void* s_LocalFree(void* p){ free(p); return 0; }
MSABI static u32   s_TlsAlloc(void){ int i=__sync_fetch_and_add(&g_tlsnext,1); return i<2048?i:0xFFFFFFFF; }
MSABI static void* s_TlsGetValue(u32 i){ g_lasterr=0; return i<2048?g_tls[i]:0; }
MSABI static int   s_TlsSetValue(u32 i,void* v){ if(i<2048){g_tls[i]=v;return 1;} return 0; }
MSABI static int   s_TlsFree(u32 i){ (void)i; return 1; }
MSABI static u32   s_FlsAlloc(void* cb){ (void)cb; return s_TlsAlloc(); }
MSABI static void* s_FlsGetValue(u32 i){ return s_TlsGetValue(i); }
MSABI static int   s_FlsSetValue(u32 i,void* v){ return s_TlsSetValue(i,v); }
MSABI static int   s_FlsFree(u32 i){ (void)i; return 1; }
MSABI static u32   s_GetCurrentThreadId(void){ return (u32)(u64)pthread_self(); }
MSABI static u32   s_GetCurrentProcessId(void){ return (u32)getpid(); }
MSABI static void* s_GetCurrentProcess(void){ return (void*)-1; }
MSABI static void* s_GetCurrentThread(void){ return (void*)-2; }
MSABI static void  s_GetSystemTimeAsFileTime(void* ft){ struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts);
    u64 v=(u64)ts.tv_sec*10000000ULL + ts.tv_nsec/100 + 116444736000000000ULL; memcpy(ft,&v,8); }
MSABI static int   s_QueryPerformanceCounter(void* x){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    u64 v=(u64)ts.tv_sec*1000000000ULL+ts.tv_nsec; memcpy(x,&v,8); return 1; }
MSABI static int   s_QueryPerformanceFrequency(void* x){ u64 v=1000000000ULL; memcpy(x,&v,8); return 1; }
MSABI static void* s_EncodePointer(void* p){ return p; }
MSABI static void* s_DecodePointer(void* p){ return p; }
MSABI static int   s_IsProcessorFeaturePresent(u32 f){ (void)f; return 1; }
MSABI static int   s_IsDebuggerPresent(void){ return 0; }
MSABI static void  s_InitializeSListHead(void* h){ memset(h,0,16); }
MSABI static void* s_InterlockedFlushSList(void* h){ (void)h; return 0; }
MSABI static u32   s_GetLastError(void){ return (u32)g_lasterr; }
MSABI static void  s_SetLastError(u32 e){ g_lasterr=(int)e; }
MSABI static void  s_noop(void){ }
MSABI static int   s_ret1(void){ return 1; }
MSABI static void* s_ret0p(void){ return 0; }
MSABI static void* s_GetModuleHandleW(void* n){ (void)n; return (void*)0x140000000ULL; }
static void* make_trap(const char* name);       // fwd
static void* g_stubs_lookup(const char* fn);     // fwd
MSABI static void* s_GetProcAddress(void* m,const char* n){ (void)m; if(!n) return 0;
    void* s=g_stubs_lookup(n); if(s) return s;
    char* nm=malloc(strlen(n)+9); sprintf(nm,"dynamic:%s",n); return make_trap(nm); }
static u64 g_hcount=0x1000;
MSABI static void* s_CreateHandle(void){ return (void*)__sync_fetch_and_add(&g_hcount,1); }  // fake non-null HANDLE
MSABI static int   s_CloseHandle(void* h){ (void)h; return 1; }
MSABI static u32   s_WaitForSingleObject(void* h,u32 ms){ (void)h;(void)ms; return 0; }       // WAIT_OBJECT_0
MSABI static u64   s_GetTickCount64(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (u64)ts.tv_sec*1000+ts.tv_nsec/1000000; }
MSABI static int   s_InitOnceExecuteOnce(void* once,void* fn,void* par,void** ctx){ (void)once;(void)par;(void)ctx;
    typedef int MSABI (*io_t)(void*,void*,void**); if(fn) ((io_t)fn)(once,par,ctx); return 1; }
MSABI static void  s_GetStartupInfoW(void* si){ if(si) memset(si,0,104); }
MSABI static void* s_GetCommandLineW(void){ static const u16 w[]={'a',0}; return (void*)w; }
MSABI static void  s_ExitProcess(u32 c){ logs("[ExitProcess]\n"); _exit((int)c); }
MSABI static void  s_TerminateProcess(void* h,u32 c){ (void)h; logs("[TerminateProcess]\n"); _exit((int)c); }
MSABI static void  s_OutputDebugStringA(const char* s){ logn("[OutputDebugStringA] ", s?s:""); }
MSABI static u32   s_GetModuleFileNameW(void* m,void* buf,u32 n){ (void)m; if(n&&buf)((u16*)buf)[0]=0; return 0; }
static u16 g_env[2]={0,0};                      // empty double-null environment block
MSABI static void* s_GetEnvironmentStringsW(void){ return g_env; }
MSABI static int   s_FreeEnvironmentStringsW(void* p){ (void)p; return 1; }
MSABI static void* s_LoadLibraryExW(void* n,void* h,u32 f){ (void)n;(void)h;(void)f; return (void*)0x140000000ULL; }
MSABI static int   s_GetModuleHandleExW(u32 f,void* n,void** out){ (void)f;(void)n; if(out)*out=(void*)0x140000000ULL; return 1; }
MSABI static u32   s_GetEnvironmentVariableA(const char* n,void* b,u32 s){ (void)n;(void)b;(void)s; g_lasterr=203; return 0; }
MSABI static int s_WideCharToMultiByte(u32 cp,u32 fl,const u16* w,int wl,char* mb,int mbl,void* dc,void* ud){
    (void)cp;(void)fl;(void)dc;(void)ud; int i=0; if(wl<0){ int n=0; while(w[n])n++; wl=n+1; }
    if(mbl==0) return wl; for(;i<wl&&i<mbl;i++) mb[i]=(char)(w[i]&0xFF); return i; }
MSABI static int s_MultiByteToWideChar(u32 cp,u32 fl,const char* mb,int mbl,u16* w,int wl){
    (void)cp;(void)fl; int i=0; if(mbl<0){ int n=0; while(mb[n])n++; mbl=n+1; }
    if(wl==0) return mbl; for(;i<mbl&&i<wl;i++) w[i]=(u8)mb[i]; return i; }

typedef void* (*resolver_t)(const char* dll, const char* fn);

struct { const char* name; void* fn; } g_stubs[] = {
    {"GetProcessHeap",s_GetProcessHeap},{"HeapAlloc",s_HeapAlloc},{"HeapReAlloc",s_HeapReAlloc},
    {"HeapFree",s_HeapFree},{"HeapSize",s_HeapSize},{"HeapCreate",s_HeapCreate},{"HeapDestroy",s_HeapDestroy},
    {"LocalAlloc",s_LocalAlloc},{"LocalFree",s_LocalFree},
    {"TlsAlloc",s_TlsAlloc},{"TlsGetValue",s_TlsGetValue},{"TlsSetValue",s_TlsSetValue},{"TlsFree",s_TlsFree},
    {"FlsAlloc",s_FlsAlloc},{"FlsGetValue",s_FlsGetValue},{"FlsSetValue",s_FlsSetValue},{"FlsFree",s_FlsFree},
    {"GetCurrentThreadId",s_GetCurrentThreadId},{"GetCurrentProcessId",s_GetCurrentProcessId},
    {"GetCurrentProcess",s_GetCurrentProcess},{"GetCurrentThread",s_GetCurrentThread},
    {"GetSystemTimeAsFileTime",s_GetSystemTimeAsFileTime},
    {"QueryPerformanceCounter",s_QueryPerformanceCounter},{"QueryPerformanceFrequency",s_QueryPerformanceFrequency},
    {"EncodePointer",s_EncodePointer},{"DecodePointer",s_DecodePointer},
    {"IsProcessorFeaturePresent",s_IsProcessorFeaturePresent},{"IsDebuggerPresent",s_IsDebuggerPresent},
    {"InitializeSListHead",s_InitializeSListHead},{"InterlockedFlushSList",s_InterlockedFlushSList},
    {"GetLastError",s_GetLastError},{"SetLastError",s_SetLastError},
    {"GetModuleHandleW",s_GetModuleHandleW},{"GetModuleHandleA",s_GetModuleHandleW},
    {"GetProcAddress",s_GetProcAddress},{"GetStartupInfoW",s_GetStartupInfoW},
    {"GetCommandLineW",s_GetCommandLineW},{"GetCommandLineA",s_GetCommandLineW},
    {"ExitProcess",s_ExitProcess},{"TerminateProcess",s_TerminateProcess},
    {"OutputDebugStringA",s_OutputDebugStringA},{"GetModuleFileNameW",s_GetModuleFileNameW},
    // benign no-ops / return-1 for sync + misc CRT init calls
    {"InitializeCriticalSection",s_noop},{"InitializeCriticalSectionEx",s_ret1},
    {"InitializeCriticalSectionAndSpinCount",s_ret1},{"EnterCriticalSection",s_noop},
    {"LeaveCriticalSection",s_noop},{"DeleteCriticalSection",s_noop},{"TryEnterCriticalSection",s_ret1},
    {"InitializeSRWLock",s_noop},{"AcquireSRWLockExclusive",s_noop},{"ReleaseSRWLockExclusive",s_noop},
    {"InitializeConditionVariable",s_noop},{"WakeConditionVariable",s_noop},{"WakeAllConditionVariable",s_noop},
    {"SetUnhandledExceptionFilter",s_ret0p},{"UnhandledExceptionFilter",s_ret1},
    {"RtlLookupFunctionEntry",s_ret0p},{"RtlPcToFileHeader",s_ret0p},
    // synchronisation / handle-returning (must be non-NULL) + CRT dynamic probes
    {"CreateEventW",s_CreateHandle},{"CreateEventExW",s_CreateHandle},{"CreateEventA",s_CreateHandle},
    {"CreateSemaphoreW",s_CreateHandle},{"CreateSemaphoreExW",s_CreateHandle},
    {"CreateMutexW",s_CreateHandle},{"CreateMutexExW",s_CreateHandle},
    {"CloseHandle",s_CloseHandle},{"WaitForSingleObject",s_WaitForSingleObject},
    {"WaitForSingleObjectEx",s_WaitForSingleObject},{"SetEvent",s_ret1},{"ResetEvent",s_ret1},
    {"GetTickCount64",s_GetTickCount64},{"GetSystemTimePreciseAsFileTime",s_GetSystemTimeAsFileTime},
    {"InitOnceExecuteOnce",s_InitOnceExecuteOnce},
    {"SleepConditionVariableCS",s_ret1},{"SleepConditionVariableSRW",s_ret1},
    {"TryAcquireSRWLockExclusive",s_ret1},{"AcquireSRWLockShared",s_noop},{"ReleaseSRWLockShared",s_noop},
    {"GetEnvironmentStringsW",s_GetEnvironmentStringsW},{"FreeEnvironmentStringsW",s_FreeEnvironmentStringsW},
    {"GetEnvironmentVariableA",s_GetEnvironmentVariableA},
    {"LoadLibraryExW",s_LoadLibraryExW},{"LoadLibraryW",s_LoadLibraryExW},{"LoadLibraryA",s_LoadLibraryExW},
    {"LoadLibraryExA",s_LoadLibraryExW},{"GetModuleHandleExW",s_GetModuleHandleExW},{"GetModuleHandleExA",s_GetModuleHandleExW},
    {"WideCharToMultiByte",s_WideCharToMultiByte},{"MultiByteToWideChar",s_MultiByteToWideChar},
    {0,0}
};
static void* g_stubs_lookup(const char* fn){ for(int i=0;g_stubs[i].name;i++) if(!strcmp(g_stubs[i].name,fn)) return g_stubs[i].fn; return 0; }

// generate an ms_abi trampoline: sub rsp,0x28; mov rcx,name; mov rax,log; call rax; add rsp,0x28; ret
static u8* g_code; static size_t g_codeoff;
MSABI static u64 trap_log(const char* name){ logn("[STUB] ", name); return 0; }
static void* make_trap(const char* name){
    u8* p = g_code + g_codeoff; u8* start = p;
    *p++=0x48;*p++=0x83;*p++=0xEC;*p++=0x28;                 // sub rsp,0x28
    *p++=0x48;*p++=0xB9; memcpy(p,&name,8); p+=8;            // mov rcx, name
    void* lg=(void*)trap_log; *p++=0x48;*p++=0xB8; memcpy(p,&lg,8); p+=8; // mov rax, trap_log
    *p++=0xFF;*p++=0xD0;                                     // call rax
    *p++=0x48;*p++=0x83;*p++=0xC4;*p++=0x28;                 // add rsp,0x28
    *p++=0xC3;                                               // ret
    g_codeoff += (size_t)(p-start);
    return start;
}
static void* resolve(const char* dll,const char* fn){
    for(int i=0;g_stubs[i].name;i++) if(!strcmp(g_stubs[i].name,fn)) return g_stubs[i].fn;
    // Everything else (incl. vulkan-1/nvcuda for now) -> logging trap.
    char* nm=malloc(strlen(dll)+strlen(fn)+2); sprintf(nm,"%s:%s",dll,fn);
    return make_trap(nm);
}

static void segv(int s,siginfo_t* si,void* uc){ (void)s;(void)uc; logs("[SIGSEGV @ ");
    char b[19]="0x0000000000000000"; u64 a=(u64)si->si_addr; for(int i=0;i<16;i++){int d=(a>>((15-i)*4))&0xF; b[2+i]=d<10?'0'+d:'a'+d-10;} (void)write(2,b,18); logs("]\n"); _exit(42); }

int main(int argc,char** argv){
    const char* path = argc>1?argv[1]:"/usr/lib/nvidia/wine/nvngx_dlssg.dll";
    int fd=open(path,O_RDONLY); if(fd<0){perror("open");return 2;} struct stat st; fstat(fd,&st);
    u8* file=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    u32 e=rd32(file+0x3C); const u8* nt=file+e; const u8* fh=nt+4; const u8* oh=fh+20;
    u16 nsec=rd16(fh+2), optsz=rd16(fh+16); u64 imgbase=rd64(oh+24);
    u32 sizeimg=rd32(oh+56), sizehdr=rd32(oh+60), ndir=rd32(oh+108); const u8* dir=oh+112;
    u32 imp=rd32(dir+8), rel=rd32(dir+40), relsz=rd32(dir+44);
    u32 tls=ndir>9?rd32(dir+9*8):0, entry=rd32(oh+16);
    printf("== S2 load: %s  sizeOfImage=%u  entryRVA=0x%x  TLSdir=%s\n",
           path,sizeimg,entry, tls?"PRESENT":"none");

    u8* base=mmap(NULL,sizeimg,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    memcpy(base,file,sizehdr);
    const u8* sec=oh+optsz;
    for(u16 i=0;i<nsec;i++){const u8* s=sec+i*40; u32 vs=rd32(s+8),va=rd32(s+12),rs=rd32(s+16),rp=rd32(s+20);
        u32 n=rs<vs?rs:vs; if(va+n<=sizeimg&&rp+n<=st.st_size) memcpy(base+va,file+rp,n);}
    int64_t delta=(int64_t)((u64)base-imgbase); u32 nrel=0;
    if(rel&&delta){const u8* p=base+rel,*end=p+relsz; while(p+8<=end){u32 pg=rd32(p),bl=rd32(p+4); if(bl<8)break;
        for(u32 i=0;i<(bl-8)/2;i++){u16 en=rd16(p+8+i*2); if((en>>12)==10){u8* t=base+pg+(en&0xFFF);u64 v=rd64(t)+delta;memcpy(t,&v,8);nrel++;}} p+=bl;}}

    g_code=mmap(NULL,1<<20,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    int nimp=0,ntrap=0;
    if(imp){const u8* d=base+imp; for(;;d+=20){u32 oft=rd32(d),nm=rd32(d+12),ft=rd32(d+16); if(!nm&&!ft&&!oft)break;
        const char* dll=(const char*)(base+nm); const u8* t=base+(oft?oft:ft); u8* iat=base+ft;
        for(;;t+=8,iat+=8){u64 v=rd64(t); if(!v)break; if(v&0x8000000000000000ULL){continue;}
            const char* fn=(const char*)(base+(u32)v+2); void* r=resolve(dll,fn);
            memcpy(iat,&r,8); nimp++; }}}
    // set executable protection on sections with MEM_EXECUTE (0x20000000), RW others
    for(u16 i=0;i<nsec;i++){const u8* s=sec+i*40; u32 va=rd32(s+12),vs=rd32(s+8),ch=rd32(s+36);
        u32 len=(vs+0xFFF)&~0xFFFu; int prot=PROT_READ; if(ch&0x80000000)prot|=PROT_WRITE; if(ch&0x20000000)prot|=PROT_EXEC;
        if(va+len<=sizeimg) mprotect(base+va,len,prot);}
    printf("   mapped@%p relocs=%u imports_wired=%u (trap code used=%zuB)\n",(void*)base,nrel,nimp,g_codeoff);
    (void)ntrap;

    // run TLS callbacks + DllMain(DLL_PROCESS_ATTACH=1) in a child under SIGSEGV guard
    printf("   forking to run DllMain(entry=0x%x)...\n\n",entry); fflush(stdout);
    pid_t pid=fork();
    if(pid==0){
        struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_sigaction=segv; sa.sa_flags=SA_SIGINFO;
        sigaction(SIGSEGV,&sa,0); sigaction(SIGILL,&sa,0); sigaction(SIGBUS,&sa,0); sigaction(SIGFPE,&sa,0);

        // --- Windows TEB via the gs segment (Linux uses fs for glibc; gs is free). The
        // DLL's CRT reads gs:[0x58] = ThreadLocalStoragePointer, gs:[0x30] = TEB self,
        // gs:[0x60] = PEB. Without this it faults at 0x58. This is the Wine TEB problem. ---
        static u8 teb[0x2000]; static u8 peb[0x800]; static void* tls_slots[512];
        memset(teb,0,sizeof teb); memset(peb,0,sizeof peb); memset(tls_slots,0,sizeof tls_slots);
        *(void**)(teb+0x30)=teb;         // NT_TIB.Self / TEB self
        *(void**)(teb+0x58)=tls_slots;   // ThreadLocalStoragePointer
        *(void**)(teb+0x60)=peb;         // ProcessEnvironmentBlock
        // give it a plausible stack range (NT_TIB StackBase/StackLimit at 0x08/0x10)
        u8* sp; __asm__("mov %%rsp,%0":"=r"(sp));
        *(void**)(teb+0x08)=sp+0x100000; *(void**)(teb+0x10)=sp-0x400000;
        // --- PE-TLS: allocate the thread's TLS block from the image template, store the
        // slot index, and hang it off tls_slots[index] so gs:[0x58][index] is valid. ---
        u64 tls_cbs=0;
        if(tls){ const u8* td=base+tls; u64 start=rd64(td+0),endr=rd64(td+8),idxaddr=rd64(td+16); tls_cbs=rd64(td+24); u32 zf=rd32(td+32);
            u64 raw=endr-start; u8* blk=calloc(1,raw+zf+64); if(raw) memcpy(blk,(void*)start,raw);
            u32 idx=1; *(u32*)idxaddr=idx; tls_slots[idx]=blk;   // slot 0 reserved
        }
        syscall(SYS_arch_prctl, 0x1001 /*ARCH_SET_GS*/, (unsigned long)teb);
        // TLS callbacks run only now that gs (the TEB) is live.
        if(tls_cbs){ void** cb=(void**)tls_cbs; for(;*cb;cb++){ typedef void MSABI(*tcb_t)(void*,u32,void*); ((tcb_t)*cb)((void*)base,1,0); } }

        typedef int MSABI (*dllmain_t)(void*,u32,void*);
        dllmain_t dm=(dllmain_t)(base+entry);
        logs("   [calling DllMain]\n");
        int r=dm((void*)base,1,0);
        char b[32]; int n=snprintf(b,sizeof b,"   [DllMain returned %d]\n",r); (void)write(2,b,n);
        _exit(r?0:7);
    }
    int status; waitpid(pid,&status,0);
    if(WIFEXITED(status)) printf("\n== child exited code %d\n",WEXITSTATUS(status));
    else if(WIFSIGNALED(status)) printf("\n== child killed by signal %d\n",WTERMSIG(status));
    printf("== S2 verdict: DllMain %s.\n",
           WIFEXITED(status)&&WEXITSTATUS(status)==0?"RETURNED 1 (DLL loaded + attached natively)":"did not complete");
    return 0;
}
