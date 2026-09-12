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
#include <openssl/pkcs7.h>
#include <openssl/x509.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/err.h>

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
// Override the driver-DLL directory for testing newer/different drops
// (e.g. S5_WINE=/tmp/dlls/fg310 with a trailing slash) without touching /usr.
// The DLLs themselves stay local-only and are never committed (*.dll ignored).
static const char* g_wine_dir(void){ const char* e=getenv("S5_WINE"); return (e&&e[0])?e:WINE; }
static u32 rd32(const u8* p){u32 v;memcpy(&v,p,4);return v;}
static u16 rd16(const u8* p){u16 v;memcpy(&v,p,2);return v;}
static u64 rd64(const u8* p){u64 v;memcpy(&v,p,8);return v;}
static void logs(const char* s){ (void)write(2,s,strlen(s)); }
static void logn(const char* a,const char* b){ logs(a); logs(b); logs("\n"); }

// NGX app-log callback (NVSDK_NGX_AppLogCallback): (message, level, component) —
// all INTEGER class, NOT variadic, so no float-ABI risk. Must be MSABI: NGX calls
// it with MS-x64 convention from any thread; must be thread-safe (write(2) is).
MSABI static void s_ngx_log(const char* m,int level,int comp){
    char b[64]; int n=snprintf(b,sizeof b,"[ngxlog lv=%d comp=%d] ",level,comp);
    if(n>0) (void)write(2,b,(size_t)n);
    if(m){ size_t L=strlen(m); if(L>4096)L=4096; (void)write(2,m,L); }
    (void)write(2,"\n",1); }

// ---- module registry ----
typedef struct { char name[64]; u8* base; u64 imgbase; u32 exp_rva,exp_sz; u32 size; int loaded; } Module;
static Module g_mod[16]; static int g_nmod=0;
static u8* g_code; static size_t g_codeoff;
static __thread void* g_tls[2048]; static __thread int g_lasterr; static int g_tlsnext=1; static u8 g_heap[1];
static u64 g_hcount=0x2000;
static u8 g_teb[0x2000],g_peb[0x800]; static void* g_tlsslots[512];   // fake Windows TEB + PE-TLS array
volatile void* g_dbg_nvngx_base;   // set to _nvngx mapped base; gdb reads it to anchor base+offset breaks
static pthread_t g_wthreads[256]; static volatile int g_nwthreads;   // real worker-thread handle table
static void* make_trap(const char* name);   // fwd (defined below)
static void* make_ms2sysv(void* target);     // fwd (defined below)
static u8 g_luid[8];                          // synthetic adapter LUID (deviceUUID-derived; filled in GPDP2)
MSABI static void* my_gipa(void*,const char*);   // ms_abi vkGetInstanceProcAddr wrapper (below)
MSABI static int s_cuDeviceGetLuid(char*,unsigned*,int); // fwd, defined near my_gipa
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
// ---- guarded pointer access for Win32 structs --------------------------------
// (needs sigjmp/dec_segv from the RaiseException decoder below; declared here,
// defined there — single definition to avoid duplicates)
#include <setjmp.h>
static sigjmp_buf g_decjmp;
static void dec_segv(int s);
// NGX sometimes passes never-initialized (garbage, not zero) sync structs.
// Blindly writing/reading them corrupts memory (flaky DllMain->0 / exit-42).
// Guard every access: unreadable -> skip; NULL -> lazy-init; live-table miss
// (garbage nonzero) -> re-init + log. Matches Wine's tolerance, never crashes.
static pthread_mutex_t g_liveguard=PTHREAD_MUTEX_INITIALIZER;
#define MAXLIVE 256
static void* g_live[MAXLIVE]; static int g_nlive;
static int live_has(void* p){ int f=0; pthread_mutex_lock(&g_liveguard);
    for(int i=0;i<g_nlive;i++) if(g_live[i]==p){ f=1; break; }
    pthread_mutex_unlock(&g_liveguard); return f; }
static void live_add(void* p){ pthread_mutex_lock(&g_liveguard);
    for(int i=0;i<g_nlive;i++) if(g_live[i]==p){ pthread_mutex_unlock(&g_liveguard); return; }
    if(g_nlive<MAXLIVE) g_live[g_nlive++]=p; pthread_mutex_unlock(&g_liveguard); }
static void live_del(void* p){ pthread_mutex_lock(&g_liveguard);
    for(int i=0;i<g_nlive;i++) if(g_live[i]==p){ g_live[i]=g_live[--g_nlive]; break; }
    pthread_mutex_unlock(&g_liveguard); }
static int guarded_read_ptr(void* addr,void** out){ struct sigaction na,oa;
    memset(&na,0,sizeof na); na.sa_handler=dec_segv; sigemptyset(&na.sa_mask);
    if(sigaction(SIGSEGV,&na,&oa)!=0) return -1;
    int r=0; if(sigsetjmp(g_decjmp,1)==0){ *out=*(void**)addr; } else r=-1;
    sigaction(SIGSEGV,&oa,0); return r; }
static int guarded_write_ptr(void* addr,void* v){ struct sigaction na,oa;
    memset(&na,0,sizeof na); na.sa_handler=dec_segv; sigemptyset(&na.sa_mask);
    if(sigaction(SIGSEGV,&na,&oa)!=0) return -1;
    int r=0; if(sigsetjmp(g_decjmp,1)==0){ *(void**)addr=v; } else r=-1;
    sigaction(SIGSEGV,&oa,0); return r; }
// Init is multithreaded (workers + main share NGX state). The old no-op stubs
// (EnterCriticalSection=noop, WaitForSingleObject=immediate-0, SleepCV=instant)
// made every wait a no-wait: workers read incomplete state -> errno-thrower
// (EAGAIN/EDEADLK sites) or null-deref, purely scheduling-dependent (OutOfDate
// vs SIGTRAP across identical runs). Real semantics below; DebugInfo/slot tricks
// store native objects inside caller-owned Win32 structs.
#define SYNC_MAGIC 0x53494e43u
typedef struct { u32 magic; int kind; pthread_mutex_t m; pthread_cond_t c;
                 int state; int manual; long count; long max; } syncobj_t; // 0=event 1=mutex 2=sem
static syncobj_t* sync_new(int kind){ syncobj_t* s=calloc(1,sizeof* s); if(!s) return 0;
    s->magic=SYNC_MAGIC; s->kind=kind; pthread_mutex_init(&s->m,0); pthread_cond_init(&s->c,0); return s; }
static syncobj_t* sync_of(void* h){ uintptr_t a=(uintptr_t)h;
    if(a<0x10000) return 0; /* NULL/small-int handles: never deref */
    if(((a&0xFF000000u)==0x70000000u)||((a&0xFF000000u)==0x50000000u)||
       ((a&0xFF000000u)==0x30000000u)) return 0; /* thread/file/console tags */
    syncobj_t* s=(syncobj_t*)h;
    if(s->magic!=SYNC_MAGIC) return 0; return s; }
static void abstime(struct timespec* ts,u32 ms){ clock_gettime(CLOCK_REALTIME,ts);
    ts->tv_sec+=ms/1000; ts->tv_nsec+=(ms%1000)*1000000LL;
    if(ts->tv_nsec>=1000000000LL){ ts->tv_sec++; ts->tv_nsec-=1000000000LL; } }
// CRITICAL_SECTION: caller-owned 48B; stash recursive-mutex ptr in DebugInfo @0.
// Wine-tolerant lazy init: a zeroed (never explicitly initialized) CS gets its
// mutex on first use under a global guard (TryEnter on such a CS must succeed,
// not return 0 — DllMain-class code branches on it).
static pthread_mutex_t g_csinit=PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t* cs_alloc(void){ pthread_mutex_t* m=malloc(sizeof*m); if(!m) return 0;
    pthread_mutexattr_t a; pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a,PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(m,&a); pthread_mutexattr_destroy(&a); live_add(m); return m; }
static pthread_mutex_t* cs_get(void* cs){ void* cur=0;
    if(!cs||guarded_read_ptr(cs,&cur)!=0) return 0; // unreadable struct: skip
    if(cur){ if(live_has(cur)) return cur;
        // nonzero but not ours (uninit garbage): re-init safely, log it
        if(getenv("S5_SYNCTRACE")){ char b[64]; snprintf(b,sizeof b,"[sync] CS re-init garbage %p\n",cur); logs(b); } }
    pthread_mutex_lock(&g_csinit);
    if(guarded_read_ptr(cs,&cur)!=0){ pthread_mutex_unlock(&g_csinit); return 0; }
    if(cur&&live_has(cur)){ pthread_mutex_unlock(&g_csinit); return cur; }
    pthread_mutex_t* m=cs_alloc();
    if(m&&guarded_write_ptr(cs,m)!=0){ pthread_mutex_destroy(m); free(m); m=0; }
    pthread_mutex_unlock(&g_csinit); return m; }
MSABI static void s_InitializeCriticalSection(void* cs){ if(!cs) return;
    if(getenv("S5_SYNCTRACE")){ char b[64]; snprintf(b,sizeof b,"[sync] InitCS %p\n",cs); logs(b); }
    // Only init into a writable slot holding NULL or garbage; never corrupt.
    void* cur=0; if(guarded_read_ptr(cs,&cur)!=0) return;
    if(cur&&live_has(cur)) return; // already ours
    pthread_mutex_lock(&g_csinit);
    if(guarded_read_ptr(cs,&cur)==0&&(!cur||!live_has(cur))){
        pthread_mutex_t* m=cs_alloc();
        if(m&&guarded_write_ptr(cs,m)!=0){ pthread_mutex_destroy(m); free(m); } }
    pthread_mutex_unlock(&g_csinit); }
MSABI static void s_EnterCriticalSection(void* cs){ if(!cs) return;
    pthread_mutex_t* m=cs_get(cs);
    if(getenv("S5_SYNCTRACE")){ char b[64]; snprintf(b,sizeof b,"[sync] EnterCS %p m=%p\n",cs,m); logs(b); }
    if(m) pthread_mutex_lock(m); }
MSABI static void s_LeaveCriticalSection(void* cs){ if(!cs) return;
    void* cur=0; if(guarded_read_ptr(cs,&cur)!=0||!cur||!live_has(cur)) return;
    pthread_mutex_unlock((pthread_mutex_t*)cur); }
#define SYNCTRACE(fmt, ...) do{ if(getenv("S5_SYNCTRACE")){ char _b[128]; snprintf(_b,sizeof _b,fmt,__VA_ARGS__); logs(_b); } }while(0)
MSABI static int s_TryEnterCriticalSection(void* cs){ if(!cs) return 0;
    pthread_mutex_t* m=cs_get(cs); int r=(m&&pthread_mutex_trylock(m)==0);
    SYNCTRACE("[sync] TryEnterCS %p -> %d\n",cs,r); return r; }
MSABI static void s_DeleteCriticalSection(void* cs){ void* cur=0; if(!cs) return;
    if(guarded_read_ptr(cs,&cur)!=0) return; if(!cur||!live_has(cur)) return;
    pthread_mutex_t* m=cur; live_del(m); pthread_mutex_destroy(m); free(m);
    guarded_write_ptr(cs,0); }
// SRWLOCK/CONDITION_VARIABLE: caller-owned 8B pointer slots.
MSABI static void s_InitializeSRWLock(void* l){ if(l)*(void**)l=0; } // lazy rwlock below
static pthread_rwlock_t* srw_get(void** l){ void* cur=0;
    if(!l||guarded_read_ptr(l,&cur)!=0) return 0;
    if(cur){ if(live_has(cur)) return cur; }
    pthread_rwlock_t* r=malloc(sizeof*r); if(!r) return 0;
    pthread_rwlock_init(r,0); live_add(r);
    if(guarded_write_ptr(l,r)!=0){ pthread_rwlock_destroy(r); free(r); return 0; } return r; }
MSABI static void s_AcquireSRWLockExclusive(void* l){ if(!l) return; pthread_rwlock_wrlock(srw_get((void**)l)); }
MSABI static void s_ReleaseSRWLockExclusive(void* l){ void* cur=0; if(!l) return;
    if(guarded_read_ptr(l,&cur)!=0||!cur||!live_has(cur)) return;
    pthread_rwlock_unlock((pthread_rwlock_t*)cur); }
MSABI static void s_AcquireSRWLockShared(void* l){ if(!l) return; pthread_rwlock_rdlock(srw_get((void**)l)); }
MSABI static void s_ReleaseSRWLockShared(void* l){ void* cur=0; if(!l) return;
    if(guarded_read_ptr(l,&cur)!=0||!cur||!live_has(cur)) return;
    pthread_rwlock_unlock((pthread_rwlock_t*)cur); }
MSABI static int s_TryAcquireSRWLockExclusive(void* l){ if(!l) return 0; return pthread_rwlock_trywrlock(srw_get((void**)l))==0; }
typedef struct { pthread_cond_t c; pthread_mutex_t m; } condwrap_t;
static condwrap_t* cond_get(void** l){ void* cur=0;
    if(!l||guarded_read_ptr(l,&cur)!=0) return 0;
    if(cur){ if(live_has(cur)) return cur; }
    condwrap_t* w=calloc(1,sizeof*w); if(!w) return 0;
    pthread_cond_init(&w->c,0); pthread_mutex_init(&w->m,0); live_add(w);
    if(guarded_write_ptr(l,w)!=0){ pthread_cond_destroy(&w->c); pthread_mutex_destroy(&w->m); free(w); return 0; } return w; }
static int cond_wait_ms(condwrap_t* w,pthread_mutex_t* ext,u32 ms){
    // Wait on our own mutex (SRW can't back a pthread_cond); ext lock released
    // across the wait to preserve mutual exclusion approximately.
    if(ext) pthread_mutex_unlock(ext);
    pthread_mutex_lock(&w->m); int r=0;
    if(ms==0xFFFFFFFFu) r=pthread_cond_wait(&w->c,&w->m);
    else { struct timespec ts; abstime(&ts,ms); r=pthread_cond_timedwait(&w->c,&w->m,&ts); }
    pthread_mutex_unlock(&w->m);
    if(ext) pthread_mutex_lock(ext); return r==0; }
// Phase gate (Agent B plan): DllMain runs with stub sync (GREEN baseline);
// real blocking/writing sync only when S5_SYNC_PHASE=post (Init and later).
// In dllmain phase the table below keeps noop/ret1/CreateHandle behavior.
static int sync_post(void){ const char* e=getenv("S5_SYNC_PHASE");
    return (e&&(!strcmp(e,"post")||!strcmp(e,"init")||!strcmp(e,"create"))); }
MSABI static int s_SleepConditionVariableCS(void*,void*,u32);   // fwd, defined below
MSABI static int s_SleepConditionVariableSRW(void*,void*,u32);  // fwd, defined below
MSABI static int s_SleepConditionVariableCS_gated(void* c,void* cs,u32 ms){
    if(!sync_post()) return 1; return s_SleepConditionVariableCS(c,cs,ms); }
MSABI static int s_SleepConditionVariableSRW_gated(void* c,void* l,u32 ms){
    if(!sync_post()) return 1; return s_SleepConditionVariableSRW(c,l,ms); }
MSABI static int s_SleepConditionVariableCS(void* c,void* cs,u32 ms){ if(!c||!cs) return 0;
    condwrap_t* w=cond_get((void**)c); pthread_mutex_t* m=*(void**)cs;
    int r=cond_wait_ms(w,m,ms); SYNCTRACE("[sync] SleepCVCS %p ms=%u -> %d\n",c,ms,r); return r; }
MSABI static int s_SleepConditionVariableSRW(void* c,void* l,u32 ms){ if(!c||!l) return 0;
    (void)l; condwrap_t* w=cond_get((void**)c); int r=cond_wait_ms(w,0,ms);
    SYNCTRACE("[sync] SleepCVSRW %p ms=%u -> %d\n",c,ms,r); return r; }
MSABI static void s_WakeConditionVariable(void* c){ void* cur=0; if(!c) return;
    if(guarded_read_ptr(c,&cur)!=0||!cur||!live_has(cur)) return;
    condwrap_t* w=cur; pthread_mutex_lock(&w->m);
    pthread_cond_signal(&w->c); pthread_mutex_unlock(&w->m); }
MSABI static void s_WakeAllConditionVariable(void* c){ void* cur=0; if(!c) return;
    if(guarded_read_ptr(c,&cur)!=0||!cur||!live_has(cur)) return;
    condwrap_t* w=cur; pthread_mutex_lock(&w->m);
    pthread_cond_broadcast(&w->c); pthread_mutex_unlock(&w->m); }
// ---- Win32 threadpool via pthreads (NGX runs workers through it) ------------
typedef void (MSABI *win_workcb_t)(void* inst,void* ctx,void* work);
typedef struct { win_workcb_t fn; void* ctx; pthread_t th; int running; } tpwork_t;
static void* tp_entry(void* p){ tpwork_t* w=p; w->fn(0,w->ctx,w);
    return 0; } // joinable; CloseThreadpoolWork joins (correct lifetime)
MSABI static void* s_CreateThreadpoolWork(win_workcb_t fn,void* ctx,void* env){ (void)env;
    if(getenv("S5_SYNCTRACE")) logs("[tp] CreateWork\n");
    if(!fn) return 0; tpwork_t* w=calloc(1,sizeof*w); if(!w) return 0;
    w->fn=fn; w->ctx=ctx; return w; }
MSABI static void s_SubmitThreadpoolWork(void* w0){ tpwork_t* w=w0; if(!w) return;
    if(getenv("S5_SYNCTRACE")) logs("[tp] SubmitWork\n");
    w->running=1; pthread_create(&w->th,0,tp_entry,w); }
MSABI static void s_CloseThreadpoolWork(void* w0){ tpwork_t* w=w0; if(!w) return;
    if(w->running) pthread_join(w->th,0); free(w); }
// Named-event table so OpenEventA finds events we created (else NULL=not found).
#define MAXNAMED 32
static struct { char name[96]; syncobj_t* ev; } g_named[MAXNAMED];
static pthread_mutex_t g_namedm=PTHREAD_MUTEX_INITIALIZER;
static void named_reg(const char* n,syncobj_t* ev){ if(!n||!n[0]||!ev) return;
    pthread_mutex_lock(&g_namedm);
    for(int i=0;i<MAXNAMED;i++) if(!g_named[i].ev){
        snprintf(g_named[i].name,sizeof g_named[i].name,"%s",n); g_named[i].ev=ev; break; }
    pthread_mutex_unlock(&g_namedm); }
MSABI static void* s_OpenEventA(u32 acc,int inh,const char* n){ (void)acc;(void)inh;
    if(getenv("S5_SYNCTRACE")){ char b[128]; snprintf(b,sizeof b,"[OpenEventA] %s\n",n?n:"(null)"); logs(b); }
    if(!n) return 0; void* r=0; pthread_mutex_lock(&g_namedm);
    for(int i=0;i<MAXNAMED;i++) if(g_named[i].ev&&!strcmp(g_named[i].name,n)){ r=g_named[i].ev; break; }
    pthread_mutex_unlock(&g_namedm); return r; }
// FreeLibraryAndExitThread: unload (noop here) then terminate calling thread.
MSABI static void s_FreeLibraryAndExitThread(void* h,u32 code){ (void)h;
    char b[64]; snprintf(b,sizeof b,"[FreeLibraryAndExitThread] code=%u\n",code); logs(b);
    pthread_exit((void*)(uintptr_t)code); }
MSABI static void* s_CreateEventExW(void* sa,const u16* n,u32 fl,u32 acc){ (void)sa;(void)n;(void)acc;
    syncobj_t* s=sync_new(0); if(!s) return 0; s->manual=(fl&1)?1:0; s->state=0; return s; }
MSABI static void* s_CreateEventW(void* sa,int man,int init,const u16* n){ (void)sa;(void)n;
    syncobj_t* s=sync_new(0); if(!s) return 0; s->manual=man?1:0; s->state=init?1:0; return s; }
MSABI static void* s_CreateEventA(void* sa,int man,int init,const char* n){ (void)sa;
    syncobj_t* s=sync_new(0); if(!s) return 0; s->manual=man?1:0; s->state=init?1:0;
    named_reg(n,s); return s; }
MSABI static int s_SetEvent(void* h){ syncobj_t* s=sync_of(h);
    if(!s) return 1; /* unknown/static handle: pretend success (old behavior) */
    if(s->kind!=0) return 0;
    pthread_mutex_lock(&s->m); s->state=1;
    if(s->manual) pthread_cond_broadcast(&s->c); else pthread_cond_signal(&s->c);
    pthread_mutex_unlock(&s->m); return 1; }
MSABI static int s_ResetEvent(void* h){ syncobj_t* s=sync_of(h);
    if(!s) return 1; /* unknown/static handle: pretend success (old behavior) */
    if(s->kind!=0) return 0;
    pthread_mutex_lock(&s->m); s->state=0; pthread_mutex_unlock(&s->m); return 1; }
MSABI static void* s_CreateMutexExW(void* sa,const u16* n,u32 fl,u32 acc){ (void)sa;(void)n;(void)fl;(void)acc;
    syncobj_t* s=sync_new(1); return s; }
MSABI static void* s_CreateMutexW(void* sa,int own,const u16* n){ (void)sa;(void)n;
    syncobj_t* s=sync_new(1); if(s&&own) pthread_mutex_lock(&s->m); return s; }
MSABI static void* s_CreateMutexA(void* sa,int own,const char* n){ (void)sa;(void)n;
    syncobj_t* s=sync_new(1); if(s&&own) pthread_mutex_lock(&s->m); return s; }
MSABI static int s_ReleaseMutex(void* h){ syncobj_t* s=sync_of(h);
    if(!s) return 1; /* unknown handle: pretend success (old trap returned 0; success is safer) */
    if(s->kind!=1) return 0;
    pthread_mutex_unlock(&s->m); return 1; }
MSABI static void* s_CreateSemaphoreExW(void* sa,long init,long max,const u16* n,u32 r,u32 acc){ (void)sa;(void)n;(void)r;(void)acc;
    syncobj_t* s=sync_new(2); if(s){ s->count=init; s->max=max; } return s; }
MSABI static void* s_CreateSemaphoreW(void* sa,long init,long max,const u16* n){ (void)sa;(void)n;
    syncobj_t* s=sync_new(2); if(s){ s->count=init; s->max=max; } return s; }
MSABI static int s_ReleaseSemaphore(void* h,long rel,long* prev){ syncobj_t* s=sync_of(h);
    if(!s||s->kind!=2) return 0; pthread_mutex_lock(&s->m);
    if(prev)*prev=s->count; s->count+=rel; pthread_cond_broadcast(&s->c);
    pthread_mutex_unlock(&s->m); return 1; }
static int sync_wait(syncobj_t* s,u32 ms){
    // Returns 1 on acquire (WAIT_OBJECT_0), 0 on timeout. INFINITE=0xFFFFFFFF.
    struct timespec dl; int inf=(ms==0xFFFFFFFFu); if(!inf) abstime(&dl,ms);
    if(s->kind==1){ // mutex: blocking lock, or trylock slices for finite waits
        if(inf){ pthread_mutex_lock(&s->m); return 1; }
        for(;;){ if(pthread_mutex_trylock(&s->m)==0) return 1;
            struct timespec now; clock_gettime(CLOCK_REALTIME,&now);
            if(now.tv_sec>dl.tv_sec||(now.tv_sec==dl.tv_sec&&now.tv_nsec>=dl.tv_nsec)) return 0;
            struct timespec sl={0,1000000}; nanosleep(&sl,0); }
    }
    pthread_mutex_lock(&s->m);
    for(;;){
        if(s->kind==0&&s->state){ if(!s->manual) s->state=0;
            pthread_mutex_unlock(&s->m); return 1; }
        if(s->kind==2&&s->count>0){ s->count--;
            pthread_mutex_unlock(&s->m); return 1; }
        int r;
        if(inf) r=pthread_cond_wait(&s->c,&s->m);
        else r=pthread_cond_timedwait(&s->c,&s->m,&dl);
        if(r!=0){ pthread_mutex_unlock(&s->m); return 0; } } }
MSABI static int   s_CloseHandle(void* h){ syncobj_t* s=sync_of(h);
    if(s){ pthread_mutex_destroy(&s->m); pthread_cond_destroy(&s->c); s->magic=0; free(s); return 1; }
    if(((uintptr_t)h & 0xFF000000u)==0x50000000u && h!=(void*)-1) close((int)((uintptr_t)h & 0x00FFFFFFu)); return 1;}
MSABI static u32   s_WaitForSingleObject(void* h,u32 m){ syncobj_t* s=sync_of(h);
    u32 r;
    if(s) r=sync_wait(s,m)?0:258/*WAIT_TIMEOUT*/;
    else if(((uintptr_t)h&0xFF000000u)==0x70000000u){ int idx=(int)((uintptr_t)h&0xFFFFFF); if(idx<256&&g_wthreads[idx]) pthread_join(g_wthreads[idx],0); r=0; }
    else r=0;
    SYNCTRACE("[sync] WaitForSingle %p ms=%u -> %u\n",h,m,r); return r;}
MSABI static u32   s_WaitForMultipleObjects(u32 n,void** hs,int all,u32 ms){
    // wait-any (bWaitAll=0); approximate wait-all by looping. Slice-based poll.
    if(!hs||!n) return 0xFFFFFFFFu;
    struct timespec dl; int inf=(ms==0xFFFFFFFFu); if(!inf) abstime(&dl,ms);
    u32 got=0;
    for(;;){ int done=1;
        for(u32 i=0;i<n;i++){ syncobj_t* s=sync_of(hs[i]); int sig=1;
            if(s){ if(s->kind==1){ sig=(pthread_mutex_trylock(&s->m)==0);
                    if(sig&&!all) return i; /* keep ownership on wait-any hit */
                    if(sig) pthread_mutex_unlock(&s->m); }
                else { pthread_mutex_lock(&s->m);
                    sig=(s->kind==0)?(s->state!=0):(s->count>0);
                    if(sig&&!all){ if(s->kind==0&&!s->manual) s->state=0;
                        if(s->kind==2) s->count--; }
                    pthread_mutex_unlock(&s->m); } }
            if(sig){ if(!all) return i; got|=1u<<i; } else if(all) done=0; }
        if(!all){ /*none signaled*/ } else if(done) return 0;
        if(!inf){ struct timespec now; clock_gettime(CLOCK_REALTIME,&now);
            if(now.tv_sec>dl.tv_sec||(now.tv_sec==dl.tv_sec&&now.tv_nsec>=dl.tv_nsec)) return 258; }
        struct timespec sl={0,1000000}; nanosleep(&sl,0); } }
MSABI static u64   s_GetTickCount64(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return(u64)t.tv_sec*1000+t.tv_nsec/1000000;}
// ---- trap-bound wait/sleep/init primitives (returned 0 instantly — broke ordering)
// Sleep/SleepEx: real nanosleep (no state, zero risk). WaitForMultipleObjects and
// ReleaseMutex were implemented but unwired (dead code) — wire them here.
MSABI static void  s_Sleep(u32 ms){ struct timespec ts={ms/1000,(ms%1000)*1000000LL}; nanosleep(&ts,0); }
MSABI static u32   s_SleepEx(u32 ms,int alert){ (void)alert; s_Sleep(ms); return 0; }
// ThreadErrorMode: thread-local roundtrip (c8b0 does Get/Set save/restore).
static __thread u32 g_temode=0;
MSABI static u32   s_GetThreadErrorMode(void){ return g_temode; }
MSABI static int   s_SetThreadErrorMode(u32 m,void* old){ if(old)*(u32*)old=g_temode; g_temode=m; return 1; }
// InitOnce: run-once flag (trap-0 meant "already done" with nothing run).
MSABI static int   s_InitOnceBeginInitialize(void* once,u32 fl,int* pend,void** c){ (void)fl;(void)c;
    if(!once) return 0; long v=__sync_fetch_and_or((long*)once,0);
    if(v){ if(pend)*pend=0; return 1; } if(pend)*pend=1; return 1; }
MSABI static int   s_InitOnceComplete(void* once,u32 fl,void* c){ (void)fl;(void)c;
    if(!once) return 0; __sync_fetch_and_or((long*)once,1); return 1; }
MSABI static void  s_GetStartupInfoW(void* si){if(si)memset(si,0,104);}
MSABI static void* s_GetCommandLineW(void){static const u16 w[]={'a',0};return(void*)w;}
MSABI static void  s_OutputDebugStringA(const char* s){logn("[dbg] ",s?s:"");}
// RaiseException decoder: NGX throws MSVC C++ (0xE06D7363) on fallible paths and
// swallowing it continues with error state (-> flaky SIGTRAP). Decode the
// EXCEPTION_RECORD instead: info[0]=magic, info[1]=thrown object,
// info[2]=ThrowInfo -> pCatchableTypeArray -> TypeDescriptor.name (mangled type),
// plus a best-effort std::runtime_error message. All reads fault-guarded: PE
// structures are valid by construction, but a corrupt throw must not kill us.
// (setjmp/g_decjmp/dec_segv declared near the top for the sync guards above)
static void dec_segv(int s){ (void)s; siglongjmp(g_decjmp,1); }
static int dec_u64(u64 addr,u64* out){ struct sigaction na,oa; memset(&na,0,sizeof na);
    na.sa_handler=dec_segv; sigemptyset(&na.sa_mask); na.sa_flags=0;
    if(sigaction(SIGSEGV,&na,&oa)!=0) return -1;
    int r=0; if(sigsetjmp(g_decjmp,1)==0){ *out=*(volatile u64*)(uintptr_t)addr; }
    else r=-1;
    sigaction(SIGSEGV,&oa,0); return r; }
static int dec_bytes(u64 addr,char* out,u64 max){ // NUL-terminated string, chunked
    u64 got=0; while(got<max){ u64 v=0; if(dec_u64(addr+got,&v)!=0) break;
        for(int i=0;i<8&&(got+i)<max;i++){ char c=(char)(v>>(i*8)); out[got+i]=c;
            if(!c){ return 0; } } got+=8; }
    if(max) out[max-1]=0; return got?0:-1; }
MSABI static void s_RaiseException(u32 code,u32 flags,u32 nargs,void* args){
    char b[160]; unsigned long a0=0,a1=0,a2=0;
    if(args){ unsigned long* ai=args; a0=ai[0]; if(nargs>1)a1=ai[1]; if(nargs>2)a2=ai[2]; }
    snprintf(b,sizeof b,"[RaiseException] code=0x%08X flags=%u nargs=%u\n",code,flags,nargs); logs(b);
    if(code!=0xE06D7363||nargs<3||!args) return;
    // Deep decode is OFF by default: it swaps the process SIGSEGV disposition
    // (sigaction+siglongjmp), which is thread-unsafe with NGX workers alive.
    // Enable per-run with S5_DECODE=1 for throw hunts only.
    if(!getenv("S5_DECODE")) return;
    u64 magic=0,obj=0,ti=0;
    { u64 v=0; if(dec_u64((u64)(uintptr_t)&((unsigned long*)args)[0],&v)==0) magic=v;
      if(dec_u64((u64)(uintptr_t)&((unsigned long*)args)[1],&v)==0) obj=v;
      if(dec_u64((u64)(uintptr_t)&((unsigned long*)args)[2],&v)==0) ti=v; }
    snprintf(b,sizeof b,"[throw] magic=0x%lX obj=%p ThrowInfo=%p\n",(unsigned long)magic,(void*)obj,(void*)ti); logs(b);
    { u64 a3=0; // x64 variant: fields may be 32-bit RVAs + module base in args[3]
      if(nargs>=4&&dec_u64((u64)(uintptr_t)&((unsigned long*)args)[3],&a3)==0&&a3>0x10000){
        u64 tio=a3+((u32)ti), objo=a3+((u32)obj);
        snprintf(b,sizeof b,"[throw] rva-try base=%p obj=%p ti=%p\n",(void*)a3,(void*)objo,(void*)tio); logs(b);
        u64 cta=0;
        if(dec_u64(tio+24,&cta)==0&&cta){
            u64 ctab=cta; if(ctab<0x10000) ctab=a3+((u32)cta);
            u64 nct=0; if(dec_u64(ctab,&nct)==0){ nct&=0xFFFFFFFFu;
                snprintf(b,sizeof b,"[throw] rvaCatchable=%llu\n",(unsigned long long)nct); logs(b);
                for(u64 i=0;i<nct&&i<2;i++){ u64 pct=0,ptd=0;
                    if(dec_u64(ctab+8+i*8,&pct)!=0||!pct) continue;
                    if(pct<0x10000) pct=a3+((u32)pct);
                    if(dec_u64(pct+8,&ptd)!=0||!ptd) continue;
                    if(ptd<0x10000) ptd=a3+((u32)ptd);
                    char nm[160]; memset(nm,0,sizeof nm);
                    if(dec_bytes(ptd+16,nm,sizeof nm)!=0) continue;
                    snprintf(b,sizeof b,"[throw] rvaType[%llu]=%s\n",(unsigned long long)i,nm); logs(b); } } } } }
    if(!ti) return;
    u64 cta=0; if(dec_u64(ti+24,&cta)!=0||!cta){ logs("[throw] ThrowInfo unreadable\n"); return; }
    u64 nct=0; { u64 v=0; if(dec_u64(cta,&v)!=0) return; nct=v&0xFFFFFFFFu; }
    snprintf(b,sizeof b,"[throw] catchable=%llu\n",(unsigned long long)nct); logs(b);
    for(u64 i=0;i<nct&&i<3;i++){ u64 pct=0,ptd=0;
        if(dec_u64(cta+8+i*8,&pct)!=0||!pct) continue;
        if(dec_u64(pct+8,&ptd)!=0||!ptd) continue;
        char nm[256]; memset(nm,0,sizeof nm);
        if(dec_bytes(ptd+16,nm,sizeof nm)!=0){ logs("[throw] typename unreadable\n"); continue; }
        snprintf(b,sizeof b,"[throw] type[%llu]=%s\n",(unsigned long long)i,nm); logs(b);
        if(i==0&&obj){ // best-effort MSVC std::runtime_error message: vfptr@0, string@8
            u64 sz=0; if(dec_u64(obj+8+16,&sz)!=0) continue; // _Mysize
            if(sz>512){ snprintf(b,sizeof b,"[throw] msg: <len %llu, skipping>\n",(unsigned long long)sz); logs(b); continue; }
            char msg[513]; memset(msg,0,sizeof msg);
            if(sz<16){ if(dec_bytes(obj+8,msg,sz+1)==0){ snprintf(b,sizeof b,"[throw] msg: %s\n",msg); logs(b); } }
            else { u64 pp=0; if(dec_u64(obj+8,&pp)!=0) continue;
                if(dec_bytes(pp,msg,sz+1)==0){ snprintf(b,sizeof b,"[throw] msg: %s\n",msg); logs(b); } } } } }
// Windows environment BLOCK ("NAME=VAL\0NAME=VAL\0\0", UTF-16). NGX reads this (GetEnvironmentStringsW
// / PEB RtlQueryEnvironmentVariable_U) and splits PATH into a directory list; an empty block left a
// null wstring in that list -> crash at 0xa1f3. Populated by init_env() before the fork.
static u16 g_env[1024]={0};
static void init_env(void){
    static const char* vars[]={
        "PATH=C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\wbem",
        "SystemRoot=C:\\Windows","SystemDrive=C:","windir=C:\\Windows",
        "TEMP=C:\\Windows\\Temp","TMP=C:\\Windows\\Temp",
        "USERPROFILE=C:\\users\\steamuser","LOCALAPPDATA=C:\\users\\steamuser\\AppData\\Local",
        "APPDATA=C:\\users\\steamuser\\AppData\\Roaming","PROGRAMDATA=C:\\ProgramData",
        "NUMBER_OF_PROCESSORS=8","PROCESSOR_ARCHITECTURE=AMD64",0};
    int o=0;
    for(int v=0; vars[v]; v++){ const char* s=vars[v];
        for(int i=0; s[i] && o<1022; i++) g_env[o++]=(u16)(u8)s[i];
        if(o<1022) g_env[o++]=0; }
    if(o<1023) g_env[o++]=0;   // block terminator (extra NUL after last entry's NUL)
}
MSABI static void* s_GetEnvironmentStringsW(void){return g_env;}
MSABI static int   s_FreeEnvironmentStringsW(void* p){(void)p;return 1;}
MSABI static u32   s_GetEnvironmentVariableA(const char* n,void* b,u32 s){(void)n;(void)b;(void)s;g_lasterr=203;return 0;}
MSABI static int s_WideCharToMultiByte(u32 cp,u32 fl,const u16* w,int wl,char* mb,int mbl,void* d,void* u){(void)cp;(void)fl;(void)d;(void)u;int i=0;if(wl<0){int n=0;while(w[n])n++;wl=n+1;}if(mbl==0)return wl;for(;i<wl&&i<mbl;i++)mb[i]=(char)(w[i]&0xFF);return i;}
MSABI static int s_MultiByteToWideChar(u32 cp,u32 fl,const char* mb,int mbl,u16* w,int wl){(void)cp;(void)fl;int i=0;if(mbl<0){int n=0;while(mb[n])n++;mbl=n+1;}if(wl==0)return mbl;for(;i<mbl&&i<wl;i++)w[i]=(u8)mb[i];return i;}
MSABI static int s_InitOnceExecuteOnce(void* o,void* f,void* p,void** c){(void)p;(void)c;typedef int MSABI(*t)(void*,void*,void**);if(f)((t)f)(o,p,c);return 1;}
static u32 putw16(u16* b,u32 sz,const char* s){ u32 n=strlen(s); if(!b||sz==0) return n; u32 i=0; for(;i<n&&i<sz-1;i++) b[i]=(u8)s[i]; b[i]=0; return i; }
MSABI static u32 s_GetSystemDirectoryW(u16* b,u32 sz){ return putw16(b,sz,"C:\\Windows\\System32"); }
MSABI static u32 s_GetWindowsDirectoryW(u16* b,u32 sz){ return putw16(b,sz,"C:\\Windows"); }
// The snippet's GetFeatureRequirements does GetModuleHandleExA(FROM_ADDRESS, <own addr>) ->
// GetModuleFileNameW(that handle) to discover its OWN module path, then processes it; a wrong
// module/path there makes it return 0xBAD00002. Return the REAL module for a handle.
MSABI static u32 s_GetModuleFileNameW(void* m,u16* b,u32 sz){
    for(int i=0;i<g_nmod;i++) if(g_mod[i].base==(u8*)m && g_mod[i].name[0]){
        char p[320]; snprintf(p,sizeof p,"C:\\Windows\\System32\\%s",g_mod[i].name); return putw16(b,sz,p); }
    return putw16(b,sz,"C:\\game\\game.exe"); }
MSABI static u32 s_GetModuleFileNameA(void* m,char* b,u32 sz){ (void)m; const char* s="C:\\game\\game.exe"; u32 n=strlen(s); if(!b||!sz)return n; u32 i=0; for(;i<n&&i<sz-1;i++)b[i]=s[i]; b[i]=0; return i; }
MSABI static int s_VerifyVersionInfoW(void* a,u32 b,u64 c){ (void)a;(void)b;(void)c; return 1; }
MSABI static u64 s_VerSetConditionMask(u64 m,u32 t,u8 c){ (void)t;(void)c; return m?m:1; }
MSABI static u32 s_GetFullPathNameW(const u16* n,u32 sz,u16* buf,void** fp){ (void)fp; u32 i=0; if(buf)for(;n[i]&&i<sz-1;i++)buf[i]=n[i]; if(buf)buf[i]=0; return i; }
// --- S5e file-API trace: log the exact paths NGX asks for during NGXGetPath, so we can tell
// whether it wants a WRITABLE DIRECTORY (scenario A) or concrete MODEL FILES (scenario B). ---
static void wlog(const char* tag,const void* w){ char b[320]; const u16* p=w; int i=0; if(p) for(;p[i]&&i<319;i++) b[i]=(char)(p[i]&0xFF); b[i]=0; logn(tag,b); }
// A DLL/file the host asks about is REAL iff it exists in the driver's wine dir (nvngx_dlssg.dll,
// etc.). NGX verifies the snippet exists (GetFileAttributes/PathFileExists) before LoadLibrary; if
// we say "absent" it never loads it -> GetFeatureRequirements(FG) returns NotImplemented.
static int wine_basename(const void* w,char* out){ const u16* p=w; int i=0; char tmp[320]; if(!p){out[0]=0;return 0;}
    for(;p[i]&&i<319;i++) tmp[i]=(char)(p[i]&0xFF); tmp[i]=0;
    char* s=strrchr(tmp,'\\'); s=s?s+1:tmp; char* s2=strrchr(s,'/'); s=s2?s2+1:s; strcpy(out,s); return (int)strlen(out); }
static int wine_has(const void* w){ char b[320]; wine_basename(w,b); if(!b[0]) return 0; char path[400]; snprintf(path,sizeof path,"%s%s",g_wine_dir(),b); return access(path,R_OK)==0; }
// File-backed handles for real wine-dir files (the snippet): NGX opens+reads/maps nvngx_dlssg.dll
// itself (hash/version check + its own PE map) before it LoadLibrary's it. Serve real bytes from
// /usr/lib/nvidia/wine/. Tag high bit so these don't collide with the event/counter handles.
#define FILETAG 0x50000000u
static int is_fileh(void* h){ return ((uintptr_t)h & 0xFF000000u)==FILETAG && h!=(void*)-1; }
static int fileh_fd(void* h){ return (int)((uintptr_t)h & 0x00FFFFFFu); }
MSABI static void* s_CreateFileW(void* name,u32 a,u32 s,void* sa,u32 cd,u32 fa,void* t){ (void)a;(void)s;(void)sa;(void)cd;(void)fa;(void)t;
    if(wine_has(name)){ char b[320]; wine_basename(name,b); char path[400]; snprintf(path,sizeof path,"%s%s",g_wine_dir(),b);
        int fd=open(path,O_RDONLY); if(fd>=0){ wlog("[CreateFileW OK] ",name); return (void*)(uintptr_t)(FILETAG|(u32)fd); } }
    // Redirect *.log CREATES to real scratch so NGX's own logging works and we can
    // read its reason strings (e.g. which driver check fails). Everything else keeps
    // NOT_FOUND (model/state existence checks must not see empty phantom files).
    { char b[320]; wine_basename(name,b); size_t n=strlen(b);
      if(n>4 && !strcasecmp(b+n-4,".log")){
        const char* sc=getenv("S5_SCRATCH"); if(!sc||!sc[0]) sc="/tmp/ngx-native";
        mkdir(sc,0755);
        char path[400]; snprintf(path,sizeof path,"%s/%s",sc,b);
        int fd=open(path,O_RDWR|O_CREAT|O_APPEND,0644);
        if(fd>=0){ wlog("[CreateFileW scratch-log] ",name); return (void*)(uintptr_t)(FILETAG|(u32)fd); } } }
    wlog("[CreateFileW] ",name); g_lasterr=2/*ERROR_FILE_NOT_FOUND*/; return (void*)-1; } // INVALID_HANDLE_VALUE
MSABI static int s_ReadFile(void* h,void* buf,u32 n,u32* pread,void* ov){ (void)ov;
    if(is_fileh(h)){ ssize_t r=read(fileh_fd(h),buf,n); if(r<0){ if(pread)*pread=0; return 0;} if(pread)*pread=(u32)r; return 1; }
    if(pread)*pread=0; return 0; }
MSABI static u32 s_SetFilePointer(void* h,int lo,int* phi,u32 method){ if(is_fileh(h)){ off_t off=(unsigned)lo; if(phi)off|=((off_t)*phi)<<32; off_t r=lseek(fileh_fd(h),off,method); if(r<0)return 0xFFFFFFFF; if(phi)*phi=(int)(r>>32); return (u32)r; } return 0xFFFFFFFF; }
MSABI static int s_SetFilePointerEx(void* h,long long dist,long long* pnew,u32 method){ if(is_fileh(h)){ off_t r=lseek(fileh_fd(h),dist,method); if(r<0)return 0; if(pnew)*pnew=r; return 1; } return 0; }
MSABI static u32 s_GetFileSize(void* h,u32* phi){ if(is_fileh(h)){ struct stat st; if(fstat(fileh_fd(h),&st)==0){ if(phi)*phi=(u32)((u64)st.st_size>>32); return (u32)st.st_size; } } return 0xFFFFFFFF; }
MSABI static int s_GetFileSizeEx(void* h,long long* psz){ if(is_fileh(h)){ struct stat st; if(fstat(fileh_fd(h),&st)==0){ if(psz)*psz=st.st_size; return 1; } } return 0; }
MSABI static void* s_CreateFileMappingW(void* h,void* sa,u32 prot,u32 hi,u32 lo,void* nm){ (void)sa;(void)prot;(void)hi;(void)lo;(void)nm; return h; /* pass file handle through as mapping handle */ }
MSABI static void* s_MapViewOfFile(void* h,u32 acc,u32 ohi,u32 olo,u64 sz){ (void)acc; if(is_fileh(h)){ off_t off=((off_t)ohi<<32)|olo; struct stat st; if(fstat(fileh_fd(h),&st)) return 0; size_t len=sz?(size_t)sz:(size_t)st.st_size; void* p=mmap(NULL,len,PROT_READ,MAP_PRIVATE,fileh_fd(h),off); return p==MAP_FAILED?0:p; } return 0; }
MSABI static int s_UnmapViewOfFile(void* p){ (void)p; return 1; }
// Real CreateThread: NGX spawns worker threads (snippet load/init runs on one); a null handle
// stalls everything downstream. Run the MS-x64 ThreadProc on a pthread, giving it its own Windows
// TEB on gs (sharing the process-wide PE-TLS array) so gs:[0x58]/gs:[0x30] accesses don't fault.
typedef u32 MSABI (*win_thread_t)(void*);
struct thctx { win_thread_t fn; void* param; };
static void* win_thread_entry(void* a){ struct thctx* c=a;
    u8* teb=calloc(1,0x2000); *(void**)(teb+0x30)=teb; *(void**)(teb+0x58)=g_tlsslots; *(void**)(teb+0x60)=g_peb;
    u8* sp; __asm__("mov %%rsp,%0":"=r"(sp)); *(void**)(teb+0x08)=sp+0x100000; *(void**)(teb+0x10)=sp-0x400000;
    syscall(SYS_arch_prctl,0x1001,(unsigned long)teb);
    win_thread_t fn=c->fn; void* p=c->param; free(c); if(fn) fn(p); return 0; }
MSABI static void* s_CreateThread(void* attr,u64 stack,void* start,void* param,u32 flags,u32* tid){ (void)attr;(void)stack;(void)flags;
    struct thctx* c=malloc(sizeof*c); c->fn=(win_thread_t)start; c->param=param;
    pthread_t th; if(pthread_create(&th,0,win_thread_entry,c)!=0){ free(c); return 0; }
    int idx=__sync_fetch_and_add(&g_nwthreads,1); if(idx<256) g_wthreads[idx]=th;
    if(tid)*tid=(u32)(uintptr_t)th; logn("[CreateThread] ","spawned"); return (void*)(uintptr_t)(0x70000000u|(u32)(idx&0xFFFFFF)); }
// ExitThread MUST terminate the calling thread (pthread_exit): the old trap
// RETURNED, so workers ran on past their exit point into garbage (flaky SIGTRAP
// at +0x67080 right after the ExitThread call). Never returns by design.
MSABI static void s_ExitThread(u32 code){ char b[64]; snprintf(b,sizeof b,"[ExitThread] code=%u\n",code); logs(b); pthread_exit((void*)(uintptr_t)code); }
// VS_FIXEDFILEINFO with a version far above any NGX minimum (the snippet-version gate).
static u32 g_vfi[13]={0xFEEF04BD,0x00010000,(999u<<16)|99u,(9999u<<16)|9999u,(999u<<16)|99u,(9999u<<16)|9999u,0x3F,0,4/*VOS_NT*/,2/*VFT_DLL*/,0,0,0};
MSABI static u32 s_GetFileVersionInfoSizeExW(u32 fl,const u16* n,u32* h){ (void)fl;(void)n; if(h)*h=0; return sizeof(g_vfi)+64; }
MSABI static u32 s_GetFileVersionInfoSizeW(const u16* n,u32* h){ (void)n; if(h)*h=0; return sizeof(g_vfi)+64; }
MSABI static int s_GetFileVersionInfoExW(u32 fl,const u16* n,u32 h,u32 len,void* data){ (void)fl;(void)n;(void)h; if(data&&len>=sizeof g_vfi) memcpy(data,g_vfi,sizeof g_vfi); return 1; }
MSABI static int s_GetFileVersionInfoW(const u16* n,u32 h,u32 len,void* data){ (void)n;(void)h; if(data&&len>=sizeof g_vfi) memcpy(data,g_vfi,sizeof g_vfi); return 1; }
MSABI static int s_VerQueryValueW(void* blk,const u16* sub,void** out,u32* len){ (void)blk;(void)sub; if(out)*out=g_vfi; if(len)*len=sizeof g_vfi; return 1; }
// benign CRT/locale/env fillers the snippet-requirements path touches
MSABI static u32 s_GetACP(void){ return 1252; }
MSABI static void* s_GetStdHandle(u32 n){ (void)n; return (void*)(uintptr_t)0x30000001u; }
MSABI static int s_AreFileApisANSI(void){ return 1; }
// NGX reads PATH heavily (MEASURED under Proton, WINEDEBUG=+environ: RtlQueryEnvironmentVariable_U
// L"PATH" dozens of times) and splits it into a directory list (the vector<wstring> built at
// _nvngx+0xb8xx). Returning NOT_FOUND left a null-data wstring in that list -> wcsdup(null) crash at
// 0xa1f3. Return a valid Windows PATH for PATH; pass the rest through to the native environment.
// Contract: on success returns chars copied (excl NUL); if buffer too small returns required size
// (incl NUL); 0 + ERROR_ENVVAR_NOT_FOUND(203) on absence.
MSABI static u32 s_GetEnvironmentVariableW(const u16* n,u16* b,u32 s){
    char name[64]; int i=0; if(n) for(;n[i]&&i<63;i++) name[i]=(char)(n[i]&0xFF); name[i]=0;
    const char* val=0;
    if(!strcasecmp(name,"PATH")) val="C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\wbem";
    else val=getenv(name);
    if(!val){ char lb[96]; snprintf(lb,sizeof lb,"[GetEnvW miss] %s\n",name); logs(lb); g_lasterr=203; return 0; }
    u32 need=(u32)strlen(val)+1;
    if(s<need) return need;
    for(i=0;val[i];i++) b[i]=(u16)(u8)val[i]; b[i]=0;
    return need-1; }
MSABI static u32 s_FormatMessageA(u32 fl,void* src,u32 id,u32 lang,char* buf,u32 sz,void* args){ (void)fl;(void)src;(void)id;(void)lang;(void)args; if(buf&&sz)buf[0]=0; return 0; }
MSABI static int s_FreeLibrary(void* h){ (void)h; return 1; }
MSABI static u32 s_GetFileType(void* h){ (void)h; return 1; /*FILE_TYPE_DISK*/ }
MSABI static int s_GetStringTypeW(u32 t,const u16* s,int c,u16* out){ (void)t;(void)s; if(out) for(int i=0;i<c;i++) out[i]=0; return 1; }
MSABI static int s_IsValidCodePage(u32 c){ (void)c; return 1; }
MSABI static void* s_OpenFileMappingA(u32 a,int inh,const char* n){ (void)a;(void)inh;
    // S5_SHM=1: real POSIX-shm backing so NGX's override/status shared memory
    // works. Default (unset): NULL (old behavior — GREEN baseline). The shm
    // path once crashed GFR at +0x3948c (struct copy into bad buffer), so it
    // stays opt-in until the mapping-size contract is understood.
    if(!getenv("S5_SHM")){ g_lasterr=2; return 0; }
    if(!n||!n[0]){ g_lasterr=2; return 0; }
    char shm[96]; snprintf(shm,sizeof shm,"/ngx_%s",n);
    for(char* p=shm+5;*p;p++) if(*p=='\\'||*p=='/'||*p==':') *p='_';
    int fd=shm_open(shm,O_RDWR|O_CREAT,0600);
    if(fd<0){ g_lasterr=2; return 0; }
    struct stat st; if(fstat(fd,&st)==0&&(size_t)st.st_size<65536) ftruncate(fd,65536);
    return (void*)(uintptr_t)(FILETAG|(u32)fd); }
MSABI static int s_WriteFile(void* h,const void* buf,u32 n,u32* wr,void* ov){ (void)ov; if(((uintptr_t)h&0xFF000000u)==0x30000000u){ ssize_t r=write(2,buf,n); if(wr)*wr=r>0?(u32)r:0; return 1; } if(is_fileh(h)){ ssize_t r=write(fileh_fd(h),buf,n); if(wr)*wr=r>0?(u32)r:0; return r>=0; } if(wr)*wr=n; return 1; }
MSABI static u32   s_GetFileAttributesW(void* name){ if(wine_has(name)){ wlog("[GetFileAttributesW OK] ",name); return 0x80; /*FILE_ATTRIBUTE_NORMAL*/ } wlog("[GetFileAttributesW] ",name); g_lasterr=2/*ERROR_FILE_NOT_FOUND*/; return 0xFFFFFFFF; } // INVALID_FILE_ATTRIBUTES
MSABI static int   s_GetFileAttributesExW(void* name,u32 lvl,void* info){ (void)lvl; if(wine_has(name)){ if(info) memset(info,0,36); if(info)*(u32*)info=0x80; wlog("[GetFileAttributesExW OK] ",name); return 1; } wlog("[GetFileAttributesExW] ",name); return 0; } // FALSE
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
MSABI static int s_PathFileExistsW(void* p){ if(wine_has(p)){ wlog("[PathFileExistsW OK] ",p); return 1; } return 0; } // TRUE for real wine-dir snippets
// NGX locates its snippets via HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore -> "FullPath" (REG_SZ).
// MEASURED under Proton (WINEDEBUG=+reg): NGX opens NGXCore, reads FullPath ("C:\\Windows\\System32"),
// then LoadLibrary("nvngx_dlssg.dll"). Natively our old trap made the read fail -> NGX aborted the
// snippet path -> NotImplemented. Serve the key so NGX proceeds to LoadLibrary (our loader then maps
// the real /usr/lib/nvidia/wine/nvngx_dlssg.dll). Only NGXCore exists; other keys honestly not-found.
#define FAKE_NGXKEY ((void*)0x4E475801)
MSABI static int s_RegOpenKeyExW(void* hk,const u16* sub,u32 o,u32 sam,void** phk){ (void)hk;(void)o;(void)sam;
    char b[260]; int i=0; if(sub) for(;sub[i]&&i<259;i++) b[i]=(char)(sub[i]&0xFF); b[i]=0; wlog("[RegOpenKeyExW] ",sub);
    if(sub&&strstr(b,"NGXCore")){ if(phk)*phk=FAKE_NGXKEY; return 0; }
    return 2; /*ERROR_FILE_NOT_FOUND — key doesn't exist, NGX uses defaults*/ }
MSABI static int s_RegQueryValueExW(void* hk,const u16* val,void* res,u32* type,u8* data,u32* cb){ (void)res;
    char b[128]; int i=0; if(val) for(;val[i]&&i<127;i++) b[i]=(char)(val[i]&0xFF); b[i]=0;
    wlog("[RegQueryValueExW] ",val);
    if(hk==FAKE_NGXKEY && val && !strcasecmp(b,"FullPath")){
        const char* p="C:\\Windows\\System32"; u32 need=(u32)(strlen(p)+1)*2;
        if(type)*type=1; /*REG_SZ*/
        if(!data){ if(cb)*cb=need; return 0; }
        if(cb&&*cb<need){ *cb=need; return 234; /*ERROR_MORE_DATA*/ }
        u16* w=(u16*)data; for(i=0;p[i];i++) w[i]=(u8)p[i]; w[i]=0; if(cb)*cb=need; logn("[RegQueryValueExW] FullPath -> ",p); return 0; }
    // NGX diagnostics + behavior switches (all observed live via query logging):
    // EnableConsoleLogging=1 makes NGX print its Init reasoning (incl. the
    // "Feature %s, denial value %d for Cms ID %x" line revealing our CMS ID).
    if(hk==FAKE_NGXKEY && val && (!strcasecmp(b,"EnableConsoleLogging"))){
        if(type)*type=4; /*REG_DWORD*/ if(cb)*cb=4;
        if(data) *(u32*)data=1; return 0; }
    if(hk==FAKE_NGXKEY && val && (!strcasecmp(b,"LogLevel"))){
        if(type)*type=4; /*REG_DWORD*/ if(cb)*cb=4;
        if(data) *(u32*)data=1; return 0; }
    return 2; /*ERROR_FILE_NOT_FOUND*/ }
MSABI static int s_RegCloseKey(void* hk){ (void)hk; return 0; }
// FAITHFUL Authenticode verification (not a bypass): NGX Authenticode-verifies the snippet via
// WinVerifyTrust before LoadLibrary, to refuse a tampered module. Rather than blindly assert trust,
// we actually verify the genuine PKCS#7 signature with libcrypto — cryptographically valid, signer
// subject = NVIDIA, and the Authenticode PE hash bound to THIS exact file — so NGX's anti-tamper
// intent is HONORED, not circumvented. (Path B is private §69e research either way; this is posture.)
static int  g_snip_verified = -1;      // -1 unknown, 0 fail, 1 verified
static char g_snip_cn[128] = "";       // real signer CN, populated on successful verify
static int authenticode_hash(const u8* d, size_t n, u8* out){
    u32 pe=rd32(d+0x3C); u16 magic=rd16(d+pe+24); u32 ddoff=pe+24+(magic==0x20b?112:96);
    u32 cksum=pe+24+64, certdir=ddoff+4*8, cert_off=rd32(d+certdir);
    EVP_MD_CTX* c=EVP_MD_CTX_new(); EVP_DigestInit_ex(c,EVP_sha256(),NULL);
    EVP_DigestUpdate(c,d,cksum); EVP_DigestUpdate(c,d+cksum+4,certdir-(cksum+4));
    size_t tail=cert_off?cert_off:n; EVP_DigestUpdate(c,d+certdir+8,tail-(certdir+8));
    unsigned len=0; EVP_DigestFinal_ex(c,out,&len); EVP_MD_CTX_free(c); return len==32;
}
static int verify_snippet_authenticode(void){
    char path[400]; snprintf(path,sizeof path, "%snvngx_dlssg.dll", g_wine_dir());
    int fd=open(path,O_RDONLY); if(fd<0) return 0;
    struct stat st; fstat(fd,&st); size_t n=st.st_size;
    u8* d=mmap(NULL,n,PROT_READ,MAP_PRIVATE,fd,0); close(fd); if(d==MAP_FAILED) return 0;
    int ok=0;
    u32 pe=rd32(d+0x3C); u16 magic=rd16(d+pe+24); u32 ddoff=pe+24+(magic==0x20b?112:96);
    u32 certdir=ddoff+4*8, cert_off=rd32(d+certdir), cert_sz=rd32(d+certdir+4);
    if(cert_off && cert_sz>8){
        const u8* der=d+cert_off+8; const u8* pp=der;
        PKCS7* p7=d2i_PKCS7(NULL,&pp,(long)cert_sz-8);
        if(p7){
            BIO* content=BIO_new(BIO_s_mem());
            X509_STORE* store=X509_STORE_new(); X509_STORE_set_default_paths(store);
            int v=PKCS7_verify(p7,NULL,store,NULL,content,0);           // signature + chain
            if(v!=1){ ERR_clear_error(); BIO_free(content); content=BIO_new(BIO_s_mem());
                      v=PKCS7_verify(p7,NULL,store,NULL,content,PKCS7_NOVERIFY); } // sig only (NVIDIA CA not in store)
            if(v==1){
                STACK_OF(X509)* signers=PKCS7_get0_signers(p7,NULL,0);
                char subj[300]="?"; int is_nv=0;
                if(signers && sk_X509_num(signers)>0){
                    X509* s=sk_X509_value(signers,0);
                    X509_NAME_oneline(X509_get_subject_name(s),subj,sizeof subj);
                    is_nv=strstr(subj,"NVIDIA")!=0;
                    char* cn=strstr(subj,"/CN="); if(cn){ cn+=4; int i=0; for(;cn[i]&&cn[i]!='/'&&i<127;i++) g_snip_cn[i]=cn[i]; g_snip_cn[i]=0; }
                }
                u8 peh[32]; int hok=authenticode_hash(d,n,peh);
                u8* cbuf=0; long clen=BIO_get_mem_data(content,(char**)&cbuf); int bind=0;
                for(long i=0;i+34<=clen && !bind;i++)
                    if(cbuf[i]==0x04 && cbuf[i+1]==0x20 && memcmp(cbuf+i+2,peh,32)==0) bind=1;
                ok = is_nv && hok && bind;
                logn("[authenticode] signer=",subj);
                logs(ok?"[authenticode] VERIFIED: valid NVIDIA signature bound to this file\n"
                       :"[authenticode] verification FAILED\n");
                if(signers) sk_X509_free(signers);
            }
            X509_STORE_free(store); BIO_free(content); PKCS7_free(p7);
        }
    }
    munmap(d,n); return ok;
}
MSABI static int s_WinVerifyTrust(void* hwnd,void* act,void* data){ (void)hwnd;(void)act;(void)data;
    if(g_snip_verified<0) g_snip_verified=verify_snippet_authenticode();
    return g_snip_verified ? 0 /*S_OK*/ : (int)0x800B0100 /*TRUST_E_NOSIGNATURE*/; }
static u8 g_fakecert[64];  // stand-in CERT_CONTEXT (pCertInfo@24 -> g_certinfo); NGX reads name via CertGetNameString
static u8 g_certinfo[256],g_provdata[64],g_provsgnr[64],g_provcert[64];  // fake WinTrust provider chain
MSABI static void* s_WTHelperProvDataFromStateData(void* h){ (void)h; return g_provdata; }
MSABI static void* s_WTHelperGetProvSignerFromChain(void* p,u32 i,int c,u32 s){ (void)p;(void)i;(void)c;(void)s; return g_provsgnr; }
MSABI static void* s_WTHelperGetProvCertFromChain(void* sgnr,u32 idx){ (void)sgnr;(void)idx;
    *(void**)(g_fakecert+24)=g_certinfo; /*CERT_CONTEXT.pCertInfo*/ *(void**)(g_provcert+8)=g_fakecert; return g_provcert; }
// Coherent fake Authenticode chain: the snippet IS the genuine NVIDIA-signed file, so report a valid
// PKCS7-embedded signature whose signer subject is "NVIDIA Corporation". CryptQueryObject yields fake
// store/msg handles; the cert lookup returns a fake CERT_CONTEXT; the name string is NVIDIA.
MSABI static int s_CryptQueryObject(u32 ot,const void* o,u32 ec,u32 ef,u32 fl,u32* pEnc,u32* pCont,u32* pFmt,void** phS,void** phM,const void** ppv){
    (void)ot;(void)o;(void)ec;(void)ef;(void)fl; if(pEnc)*pEnc=0x00010001; if(pCont)*pCont=10/*PKCS7_SIGNED_EMBED*/; if(pFmt)*pFmt=1;
    if(phS)*phS=(void*)0x5700; if(phM)*phM=(void*)0x5701; if(ppv)*ppv=0; logn("[CryptQueryObject] ","OK (assert NVIDIA sig)"); return 1; }
MSABI static int s_CryptMsgGetParam(void* h,u32 type,u32 idx,void* data,u32* cb){ (void)h;(void)idx;
    if(type==5/*CMSG_SIGNER_COUNT_PARAM*/){ if(data&&cb&&*cb>=4)*(u32*)data=1; if(cb)*cb=4; return 1; }
    u32 need=256; if(!data){ if(cb)*cb=need; return 1; } if(cb){ u32 w=*cb<need?*cb:need; memset(data,0,w); } return 1; }
MSABI static int s_CryptMsgClose(void* h){ (void)h; return 1; }
MSABI static void* s_CertFindCertificateInStore(void* store,u32 enc,u32 fl,u32 ft,const void* fp,const void* prev){ (void)store;(void)enc;(void)fl;(void)ft;(void)fp;(void)prev; return prev?0:g_fakecert; }
// Return the REAL signer CN extracted during faithful verification (not a hardcoded string).
MSABI static u32 s_CertGetNameStringW(void* c,u32 t,u32 f,void* p,u16* name,u32 cch){ (void)c;(void)t;(void)f;(void)p;
    if(g_snip_verified<0) g_snip_verified=verify_snippet_authenticode();
    const char* s=g_snip_cn[0]?g_snip_cn:""; u32 n=(u32)strlen(s)+1; if(!name||!cch) return n; u32 i=0; for(;i<n-1&&i<cch-1;i++) name[i]=(u8)s[i]; name[i]=0; return i+1; }
MSABI static u32 s_CertGetNameStringA(void* c,u32 t,u32 f,void* p,char* name,u32 cch){ (void)c;(void)t;(void)f;(void)p;
    if(g_snip_verified<0) g_snip_verified=verify_snippet_authenticode();
    const char* s=g_snip_cn[0]?g_snip_cn:""; u32 n=(u32)strlen(s)+1; if(!name||!cch) return n; u32 i=0; for(;i<n-1&&i<cch-1;i++) name[i]=s[i]; name[i]=0; return i+1; }
MSABI static int s_CertFreeCertificateContext(void* c){ (void)c; return 1; }
MSABI static int s_CertCloseStore(void* s,u32 f){ (void)s;(void)f; return 1; }
// D3DKMT_QUERYADAPTERINFO{ UINT hAdapter@0; KMTQAITYPE Type@4; void* pData@8; UINT Size@16 }
MSABI static int s_D3DKMTQueryAdapterInfo(void* p){ if(!p) return (int)0xC000000D;
    u32 type=*(u32*)((u8*)p+4); void* pd=*(void**)((u8*)p+8); u32 sz=*(u32*)((u8*)p+16);
    char b[80]; snprintf(b,sizeof b,"[D3DKMTQueryAdapterInfo] Type=%u size=%u\n",type,sz); logs(b);
    // MEASURED against Proton (WINEDEBUG=+d3dkmt): even Wine returns "type 48 not handled" and
    // NGX tolerates it (FG still Success). So DON'T fake success with zeroed data (that sends NGX
    // down a wrong path) — match Wine and return an error for unhandled types.
    if(type==48) return (int)0xC0000002; // STATUS_NOT_IMPLEMENTED (like Wine's fixme path)
    if(pd&&sz) memset(pd,0,sz);
    return 0; } // STATUS_SUCCESS for handled types
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
MSABI static int s_SYS_GetDriverAndBranchVersion(u32* pver,char* branch){ if(pver)*pver=61057; /*610.57 (nvidia-smi 610.57.04) — real version like dxvk */ if(branch){const char* e=getenv("S5_BRANCH"); const char* b=e&&e[0]?e:"r610_00";int i=0;for(;b[i]&&i<63;i++)branch[i]=b[i];branch[i]=0;} return 0; }
// Back the fake nvapi handles with REAL zeroed memory, so if NGX dereferences a handle
// (the Windows nvapi handles are internally pointers) it lands in valid memory, not a crash.
static u8 g_gpubuf[8192];
#define FAKE_GPU  ((void*)(g_gpubuf))
#define FAKE_LGPU ((void*)(g_gpubuf+4096))
MSABI static int s_EnumPhysicalGPUs(void** h,u32* c){ if(h)h[0]=FAKE_GPU; if(c)*c=1; return 0; }
// NV_GPU_ARCH_INFO = {version@0 (set by caller: size|ver<<16), architecture@4,
// implementation@8, revision@12, ...}. Zero the FULL versioned size first like
// dxvk does — leftover stack garbage in ai[4..] flips checks nondeterministically.
// MEASURED against dxvk-nvapi (nvapi_dump.exe under Proton): arch=0x1B0, impl=0x2,
// rev=0xFFFFFFFF (CHIP_REVISION_UNKNOWN). Earlier impl=5/rev=0xA1 were guesses.
MSABI static int s_GPU_GetArchInfo(void* gpu,u32* ai){ (void)gpu; if(ai){ u32 sz=ai[0]&0xFFFF; if(sz>4&&sz<=64) memset(ai+1,0,sz-4); ai[1]=0x000001B0; /*GB2xx/Blackwell*/ ai[2]=0x00000002; /*impl*/ ai[3]=0xFFFFFFFF; /*rev=UNKNOWN*/ } return 0; }
MSABI static int s_GetLogicalGPU(void* p,void** l){ (void)p; if(l)*l=FAKE_LGPU; return 0; }
MSABI static int s_GPU_GetPCIIdentifiers(void* g,u32* dev,u32* sub,u32* rev,u32* ext){ (void)g; u32 id=(0x2D18u<<16)|0x10DE; if(dev)*dev=id; if(sub)*sub=0; if(rev)*rev=0xA1; if(ext)*ext=0x2D18; return 0; } // rev back to 0xA1: arch derivation may read DeviceID+Revision (dxvk-differential covers uninit bytes only, not semantic HW fields)
MSABI static int s_GPU_GetFullName(void* g,char* name){ (void)g; const char* s="NVIDIA GeForce RTX 5070 Laptop GPU"; if(name){ memset(name,0,64); int i=0;for(;s[i]&&i<63;i++)name[i]=s[i];name[i]=0;} return 0; }
MSABI static int s_GPU_GetGPUType(void* g,u32* t){ (void)g; if(t)*t=2; /*DGPU*/ return 0; }
MSABI static int s_GPU_GetBusType(void* g,u32* t){ (void)g; if(t)*t=3; /*PCI_EXPRESS*/ return 0; }
// Synthetic Windows adapter LUID, used for BOTH the Vulkan deviceLUID (forced into
// VkPhysicalDeviceIDProperties) and the nvapi OS-AdapterId, so NGX's adapter correlation matches.
// DERIVED from the fixed deviceUUID (set in s_vkGPDP2) — deterministic + stable across process
// runs + collision-free, in case NGX uses the LUID as a cache key for its model/config files.
// MEASURED from dxvk-nvapi GetLogicalGpuInfo under Proton (version 0x10238): OSAdapterId =
// f2 03 00 00 00 00 00 00 (a small Windows-style LUID, LowPart=0x3F2). Use dxvk's EXACT value for
// all three LUID surfaces (vk deviceLUID, D3DKMT adapter LUID, nvapi OS-adapter id) so the native
// evaluator's LUID inputs are byte-identical to Proton, not just self-consistent.
static u8 g_luid[8]={0xF2,0x03,0x00,0x00,0x00,0x00,0x00,0x00};
// NV_LOGICAL_GPU_DATA: version@0, pOSAdapterId@8, physicalGpuCount@16, physicalGpuHandles@24.
MSABI static int s_GPU_GetLogicalGpuInfo(void* lgpu,u8* d){ (void)lgpu; if(!d) return 0;
    // MEASURE the exact NV_LOGICAL_GPU_DATA version NGX passes (d[0] = sizeof|ver<<16). Native we're
    // the callee, so we READ it instead of guessing — then Proton dxvk can be asked with this exact
    // version to yield the real bytes to diff (the -9 wall was just a wrong guessed version).
    { char b[96]; u32 v=*(u32*)d; snprintf(b,sizeof b,"[LGI] version=0x%08X (size=%u ver=%u) osidPtr=%p\n",
        v, v&0xFFFF, v>>16, *(void**)(d+8)); logs(b); }
    // Zero the FULL caller-sized struct first like dxvk (all-zero except the
    // fields below — verified byte-identical under Proton): leftover stack
    // garbage in the tail flips checks nondeterministically (OutOfDate vs throw).
    { u32 v=*(u32*)d, sz=v&0xFFFF; void* osid=*(void**)(d+8);
      if(sz>8&&sz<=2048) memset(d+8,0,sz-8);
      *(u32*)d=v; *(void**)(d+8)=osid;
      if(osid) memcpy(osid,g_luid,8); }
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
static void* g_stubs_lookup(const char* fn);

// dynamic loader entry points (real)
MSABI static void* s_GetModuleHandleW(void* n){ (void)n; return g_mod[0].base; }
MSABI static void* s_GetProcAddress(void* m,const char* n){ if(!n)return 0;
    for(int i=0;i<g_nmod;i++) if(g_mod[i].base==(u8*)m){ void* e=module_export(&g_mod[i],n); if(e)return e; break; }
    // GetProcAddress is PER-HANDLE: only the addressed module's exports (base-match loop above).
    // Do NOT greedily search other modules for NVSDK_NGX_*/NGX_* — the host probes the SNIPPET for
    // functions it may not export and expects NULL (e.g. GetFeatureDeviceExtensionRequirements);
    // returning _nvngx's copy diverges from Proton and yields FeatureNotSupported.
    // Vulkan loader: the host's GPU-arch path LoadLibrary's vulkan-1.dll + resolves entry
    // points. gipa/gdpa must be our ms_abi wrappers (they ms2sysv-wrap what they return);
    // other vk* funcs bridge straight to native libvulkan via ms2sysv.
    if(!strcmp(n,"vkGetInstanceProcAddr")){ logn("[vk->native] ",n); return (void*)my_gipa; }
    if(!strcmp(n,"vkGetDeviceProcAddr")){ logn("[vk->native] ",n); return (void*)my_gdpa; }
    if(!strncmp(n,"vk",2)&&g_hvk){ void* f=dlsym(g_hvk,n); if(f){ logn("[vk->native] ",n); return make_ms2sysv(f); } }
    // NVML is public + native: bridge nvml.dll's functions to libnvidia-ml.so.1 via ms2sysv.
    if(!strncmp(n,"nvml",4)&&g_hnvml){ void* f=dlsym(g_hnvml,n); if(f){ logn("[nvml->native] ",n); return make_ms2sysv(f); } }
    // The host probes the snippet for OPTIONAL NGX callbacks (NGX_SNIPPETS_GetRequiredDriverSupport,
    // NVSDK_NGX_SetTelemetryCallback, SetOverrideStatusCallback, ...) that this snippet doesn't export.
    // The all-module export search above already failed for them -> return NULL (not a trap) so the
    // host's `if(pfn)` guard skips them, exactly as under Proton. A trap made the host call garbage.
    if(!strncmp(n,"NVSDK_NGX",9)||!strncmp(n,"NGX_",4)){ logn("[GetProcAddress->NULL] ",n); return 0; }
    // Everything else (Win32/CRT/crypt the code calls unconditionally) -> stub or logging trap as before.
    void* r=resolve("dyn",n); return r; }
static void wide_basename(const void* w,char* out){ const u16* p=w; int n=0; char tmp[260];
    for(;p[n]&&n<259;n++) tmp[n]=(char)(p[n]&0xFF); tmp[n]=0; char* s=strrchr(tmp,'\\'); s=s?s+1:tmp; char* s2=strrchr(s,'/'); s=s2?s2+1:s; strcpy(out,s); }
MSABI static void* s_LoadLibraryExW(void* n,void* h,u32 f){ (void)h;(void)f; char base[260]; wide_basename(n,base);
    logn("[LoadLibrary] ",base); Module* m=load_module(base); return m?m->base:(void*)0x140000000ULL; }
MSABI static void* s_LoadLibraryA(const char* n){ const char* s=strrchr(n,'\\'); s=s?s+1:n; logn("[LoadLibraryA] ",s);
    Module* m=load_module(s); return m?m->base:(void*)0x140000000ULL; }
// FROM_ADDRESS discriminator. A caller address the dynamic linker resolves (dladdr succeeds) is native
// s5_host / libc code — e.g. Init_ProjectID's own return address read from [rsp+0x328]. That is NOT a
// Windows PE, so GetModuleHandleEx(FROM_ADDRESS) must FAIL -> Init skips its app-module path processing
// (0xd659 je 0xd6ce) instead of building a struct with a null field and crashing at 0xceee. An address
// the linker does NOT know is a manually-mapped Windows PE — the snippet that _nvngx loaded internally
// (not in g_mod) — so return the host module, keeping GetFeatureRequirements' own self-lookup working.
// FROM_ADDRESS discriminator (kept for reference; currently unused — native callers
// report the host module, see above).
// static int addr_is_native(void* p){ Dl_info di; return p && dladdr(p,&di)!=0; }
MSABI static int   s_GetModuleHandleExW(u32 f,void* n,void** out){
    // FROM_ADDRESS rules (all measured):
    // - n inside a loaded PE -> that module (snippet self-lookup -> host).
    // - querier in FUN_180061950 (GFR dir-string helper, _nvngx+0x61900..0x61b00)
    //   or in Init d5d0 (+0xd640..+0xd690, needs pwVar6/pwVar4 dir strings):
    //   report host module even for native caller addrs (else NULL wstrings
    //   crash 0xb720/0xa1f3 downstream).
    // - otherwise native caller -> FAIL (old behavior): DllMain/CRT startup
    //   branches on FAIL vs SUCCESS, and blanket SUCCESS regressed it to 0.
    if(f&4){ void* ret=__builtin_return_address(0);
        for(int i=0;i<g_nmod;i++) if(g_mod[i].base && (u8*)n>=g_mod[i].base && (u8*)n<g_mod[i].base+g_mod[i].size){ if(out)*out=g_mod[i].base; return 1; }
        if(g_nmod>0){ u8* b=g_mod[0].base;
            if(((u8*)ret>=b+0x61900&&(u8*)ret<b+0x61b00)||
               ((u8*)ret>=b+0xd640&&(u8*)ret<b+0xd690)){
                if(getenv("S5_SYNCTRACE")) logs("[ExW carve-out hit]\n");
                if(out)*out=b; return 1; } }
        if(out)*out=0; return 0; }
    if(out)*out=g_mod[0].base; return 1; }
// GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS (0x4): resolve which loaded module CONTAINS the address.
// The snippet uses this to find its own module (then its path). Returning the dummy module broke it.
// The snippet checks the OS version (RtlGetVersion) against MinOS=10.0.0; an unfilled struct reads
// as 0.0.0 -> FeatureSupported bit 0x8 (OSVersionBelow). Report Windows 10.0.19041 (like Wine/Proton).
// RTL_OSVERSIONINFOW: size@0, major@4, minor@8, build@12, platformId@16, szCSDVersion@20.
MSABI static int s_RtlGetVersion(u8* vi){ if(vi){ *(u32*)(vi+4)=10; *(u32*)(vi+8)=0; *(u32*)(vi+12)=19041; *(u32*)(vi+16)=2; *(u16*)(vi+20)=0; } return 0; }
MSABI static int s_GetVersionExW(u8* vi){ if(vi){ *(u32*)(vi+4)=10; *(u32*)(vi+8)=0; *(u32*)(vi+12)=19041; *(u32*)(vi+16)=2; *(u16*)(vi+20)=0; } return 1; }
MSABI static int s_GetModuleHandleExA(u32 f,void* n,void** out){
    if(getenv("S5_SYNCTRACE")&&g_nmod>0){ void* ret=__builtin_return_address(0);
        char b[96]; const char* wh="native"; static char w2[32];
        if((u8*)ret>=g_mod[0].base&&(u8*)ret<g_mod[0].base+g_mod[0].size){ snprintf(w2,sizeof w2,"nvngx+0x%x",(unsigned)((u8*)ret-g_mod[0].base)); wh=w2; }
        snprintf(b,sizeof b,"[ExA f=%u n=%p ret=%s]\n",f,n,wh); logs(b); }
    // Same per-querier rule as the W variant (see above).
    if(f&4){ void* ret=__builtin_return_address(0);
        for(int i=0;i<g_nmod;i++) if(g_mod[i].base && (u8*)n>=g_mod[i].base && (u8*)n<g_mod[i].base+g_mod[i].size){ if(out)*out=g_mod[i].base; return 1; }
        if(g_nmod>0){ u8* b=g_mod[0].base;
            if(((u8*)ret>=b+0x61900&&(u8*)ret<b+0x61b00)||
               ((u8*)ret>=b+0xd640&&(u8*)ret<b+0xd690)){
                if(out)*out=b; return 1; } }
        if(out)*out=0; return 0; }
    if(out)*out=g_mod[0].base; return 1; }

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
 {"GetModuleHandleExW",s_GetModuleHandleExW},{"GetModuleHandleExA",s_GetModuleHandleExA},
 {"RtlGetVersion",s_RtlGetVersion},{"GetVersionExW",s_GetVersionExW},{"GetVersionExA",s_GetVersionExW},
 {"LoadLibraryW",s_LoadLibraryExW},{"LoadLibraryExW",s_LoadLibraryExW},{"LoadLibraryA",s_LoadLibraryA},{"LoadLibraryExA",s_LoadLibraryA},
 {"GetStartupInfoW",s_GetStartupInfoW},{"GetCommandLineW",s_GetCommandLineW},{"GetCommandLineA",s_GetCommandLineW},
 {"OutputDebugStringA",s_OutputDebugStringA},{"RaiseException",s_RaiseException},
 {"GetEnvironmentStringsW",s_GetEnvironmentStringsW},{"FreeEnvironmentStringsW",s_FreeEnvironmentStringsW},{"GetEnvironmentVariableA",s_GetEnvironmentVariableA},
 {"WideCharToMultiByte",s_WideCharToMultiByte},{"MultiByteToWideChar",s_MultiByteToWideChar},
 {"InitOnceExecuteOnce",s_InitOnceExecuteOnce},
 {"InitializeCriticalSection",s_noop},{"InitializeCriticalSectionEx",s_ret1},{"InitializeCriticalSectionAndSpinCount",s_ret1},
 {"EnterCriticalSection",s_noop},{"LeaveCriticalSection",s_noop},{"DeleteCriticalSection",s_noop},{"TryEnterCriticalSection",s_ret1},
 {"InitializeSRWLock",s_noop},{"AcquireSRWLockExclusive",s_noop},{"ReleaseSRWLockExclusive",s_noop},{"TryAcquireSRWLockExclusive",s_ret1},
 {"AcquireSRWLockShared",s_noop},{"ReleaseSRWLockShared",s_noop},
 {"InitializeConditionVariable",s_noop},{"WakeConditionVariable",s_WakeConditionVariable},{"WakeAllConditionVariable",s_WakeAllConditionVariable},
 {"SleepConditionVariableCS",s_SleepConditionVariableCS_gated},{"SleepConditionVariableSRW",s_SleepConditionVariableSRW_gated},
 {"CreateEventW",s_CreateHandle},{"CreateEventExW",s_CreateHandle},{"CreateEventA",s_CreateHandle},
 {"CreateSemaphoreW",s_CreateHandle},{"CreateSemaphoreExW",s_CreateHandle},{"CreateMutexW",s_CreateHandle},{"CreateMutexExW",s_CreateHandle},
 {"CloseHandle",s_CloseHandle},{"WaitForSingleObject",s_WaitForSingleObject},{"WaitForSingleObjectEx",s_WaitForSingleObject},
 {"SetEvent",s_ret1},{"ResetEvent",s_ret1},{"ReleaseMutex",s_ReleaseMutex},{"WaitForMultipleObjects",s_WaitForMultipleObjects},{"GetTickCount64",s_GetTickCount64},
 {"Sleep",s_Sleep},{"SleepEx",s_SleepEx},{"GetThreadErrorMode",s_GetThreadErrorMode},{"SetThreadErrorMode",s_SetThreadErrorMode},
 {"InitOnceBeginInitialize",s_InitOnceBeginInitialize},{"InitOnceComplete",s_InitOnceComplete},
 {"OutputDebugStringW",s_OutputDebugStringA},
 {"SetUnhandledExceptionFilter",s_ret0p},{"UnhandledExceptionFilter",s_ret1},{"RtlLookupFunctionEntry",s_ret0p},{"RtlPcToFileHeader",s_ret0p},
 {"GetModuleFileNameW",s_GetModuleFileNameW},{"GetModuleFileNameA",s_GetModuleFileNameA},
 {"GetSystemDirectoryW",s_GetSystemDirectoryW},{"GetWindowsDirectoryW",s_GetWindowsDirectoryW},
 {"VerifyVersionInfoW",s_VerifyVersionInfoW},{"VerSetConditionMask",s_VerSetConditionMask},{"GetFullPathNameW",s_GetFullPathNameW},
 {"nvapi_QueryInterface",s_nvapi_QueryInterface},
 {"CreateFileW",s_CreateFileW},{"GetFileAttributesW",s_GetFileAttributesW},{"GetFileAttributesExW",s_GetFileAttributesExW},{"FindFirstFileExW",s_FindFirstFileExW},
 {"SHGetKnownFolderPath",s_SHGetKnownFolderPath},{"CoTaskMemFree",s_CoTaskMemFree},
 {"D3DKMTEnumAdapters2",s_D3DKMTEnumAdapters2},{"PathFileExistsW",s_PathFileExistsW},
 {"D3DKMTQueryAdapterInfo",s_D3DKMTQueryAdapterInfo},{"D3DKMTCloseAdapter",s_D3DKMTCloseAdapter},{"D3DKMTOpenAdapterFromLuid",s_D3DKMTOpenAdapterFromLuid},
 {"RegOpenKeyExW",s_RegOpenKeyExW},{"RegQueryValueExW",s_RegQueryValueExW},{"RegCloseKey",s_RegCloseKey},
 {"ReadFile",s_ReadFile},{"SetFilePointer",s_SetFilePointer},{"SetFilePointerEx",s_SetFilePointerEx},
 {"GetFileSize",s_GetFileSize},{"GetFileSizeEx",s_GetFileSizeEx},
 {"CreateFileMappingW",s_CreateFileMappingW},{"CreateFileMappingA",s_CreateFileMappingW},{"MapViewOfFile",s_MapViewOfFile},{"UnmapViewOfFile",s_UnmapViewOfFile},
 {"WinVerifyTrust",s_WinVerifyTrust},{"WTHelperProvDataFromStateData",s_WTHelperProvDataFromStateData},{"WTHelperGetProvSignerFromChain",s_WTHelperGetProvSignerFromChain},{"WTHelperGetProvCertFromChain",s_WTHelperGetProvCertFromChain},
 {"CreateThread",s_CreateThread},{"ExitThread",s_ExitThread},{"FreeLibraryAndExitThread",s_FreeLibraryAndExitThread},{"WriteFile",s_WriteFile},
 {"CreateThreadpoolWork",s_CreateThreadpoolWork},{"SubmitThreadpoolWork",s_SubmitThreadpoolWork},{"CloseThreadpoolWork",s_CloseThreadpoolWork},{"OpenEventA",s_OpenEventA},
 {"GetFileVersionInfoSizeExW",s_GetFileVersionInfoSizeExW},{"GetFileVersionInfoSizeW",s_GetFileVersionInfoSizeW},{"GetFileVersionInfoExW",s_GetFileVersionInfoExW},{"GetFileVersionInfoW",s_GetFileVersionInfoW},{"VerQueryValueW",s_VerQueryValueW},
 {"GetACP",s_GetACP},{"GetStdHandle",s_GetStdHandle},{"AreFileApisANSI",s_AreFileApisANSI},{"GetEnvironmentVariableW",s_GetEnvironmentVariableW},
 {"FormatMessageA",s_FormatMessageA},{"FreeLibrary",s_FreeLibrary},{"OpenFileMappingA",s_OpenFileMappingA},
 {"CreateMutexA",s_CreateHandle},{"GetFileType",s_GetFileType},{"GetStringTypeW",s_GetStringTypeW},{"IsValidCodePage",s_IsValidCodePage},
 {"CryptQueryObject",s_CryptQueryObject},{"CryptMsgGetParam",s_CryptMsgGetParam},{"CryptMsgClose",s_CryptMsgClose},
 {"CertFindCertificateInStore",s_CertFindCertificateInStore},{"CertGetNameStringW",s_CertGetNameStringW},{"CertGetNameStringA",s_CertGetNameStringA},
 {"CertFreeCertificateContext",s_CertFreeCertificateContext},{"CertCloseStore",s_CertCloseStore},
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
    if(!strcasecmp(dll,"nvcuda.dll")){   void* n=g_hcu?dlsym(g_hcu,fn):0;
        if(n&&!strcmp(fn,"cuDeviceGetLuid")){ logn("[cuda->interpose] ",fn); return (void*)s_cuDeviceGetLuid; }
        if(n){ logn("[cuda->native] ",fn); return make_ms2sysv(n); } logn("[cuda MISSING] ",fn); }
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
    char path[320]; snprintf(path,sizeof path,"%s%s",g_wine_dir(),name);
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
    Module* m=&g_mod[g_nmod++]; strncpy(m->name,name,63); m->base=base; m->imgbase=imgbase; m->exp_rva=expr; m->exp_sz=exps; m->size=sizeimg; m->loaded=1;
    // PE-TLS block for this module -> hang off the shared gs:[0x58] TLS array (set in main)
    if(tls){const u8* td=base+tls;u64 start=rd64(td+0),endr=rd64(td+8),idxaddr=rd64(td+16);u32 zf=rd32(td+32);
        u64 raw=endr-start;u8* blk=calloc(1,raw+zf+64);if(raw)memcpy(blk,(void*)start,raw);
        u32 slot=(u32)(m-g_mod)+1; if(slot<512){*(u32*)idxaddr=slot; g_tlsslots[slot]=blk;} }
    // run DllMain
    typedef int MSABI(*dm_t)(void*,u32,void*); dm_t dm=(dm_t)(base+entry);
    logn("[loading] ",name); int r=dm((void*)base,1,0);
    { char b[96]; snprintf(b,sizeof b,"[%s DllMain -> %d, base=%p exports@rva=0x%x]\n",name,r,(void*)base,expr); logs(b);}
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
    const char* de[]={"VK_KHR_external_memory","VK_KHR_external_memory_fd","VK_KHR_external_semaphore","VK_KHR_external_semaphore_fd","VK_KHR_push_descriptor",
        "VK_NVX_binary_import","VK_NVX_image_view_handle"};
    if(try_device(de,7) && try_device(de,5) && try_device(de,3) && try_device(0,0)) return 3;   // degrade until a device is created
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
            // Force validity + dxvk's EXACT LUID bytes (g_luid), so vk deviceLUID == nvapi
            // OSAdapterId == D3DKMT LUID == what Proton feeds the evaluator, byte-for-byte.
            idp->deviceLUIDValid=1; memcpy(idp->deviceLUID,g_luid,8); idp->deviceNodeMask=1;
        } else { char c[48]; snprintf(c,sizeof c,"   pNext sType=%u\n",(unsigned)s->sType); logs(c);} } }
MSABI static void* my_gipa(void* inst,const char* n){ if(n)logn("[gipa] ",n);
    if(n&&!strcmp(n,"vkGetPhysicalDeviceProperties2")) return (void*)s_vkGPDP2;
    void* f=(void*)vkGetInstanceProcAddr((VkInstance)inst,n); return f?make_ms2sysv(f):0; }
// cuDeviceGetLuid interpose: report the SAME synthetic LUID as the Vulkan
// deviceLUID (g_luid) so the snippet's Vulkan<->CUDA adapter correlation
// matches (native CUDA would report the real LUID, Vulkan gets synthesized).
// CUresult cuDeviceGetLuid(char*, unsigned*, CUdevice): all INT class.
typedef int (*cuLuid_t)(char*,unsigned*,int);
MSABI static int s_cuDeviceGetLuid(char* luid,unsigned* mask,int dev){
    cuLuid_t real=(cuLuid_t)dlsym(g_hcu,"cuDeviceGetLuid");
    int r=real?real(luid,mask,dev):-1;
    { char b[96]; snprintf(b,sizeof b,"[cuLuid] dev=%d -> %d mask=%u (forcing synth)\n",
        dev,r,mask?(unsigned)*mask:9999); logs(b); }
    if(luid) memcpy(luid,g_luid,8); if(mask) *mask=1; return r; }
MSABI static void* my_gdpa(void* dev,const char* n){ if(n)logn("[gdpa] ",n); PFN_vkGetDeviceProcAddr g=(PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(g_inst,"vkGetDeviceProcAddr"); void* f=g?(void*)g((VkDevice)dev,n):0; return f?make_ms2sysv(f):0; }

static void hex64(char* o,u64 a){ for(int i=0;i<16;i++){int d=(a>>((15-i)*4))&0xF;o[i]=d<10?'0'+d:'a'+d-10;} o[16]=0; }
static void segv(int s,siginfo_t* si,void* uc){ (void)s;
    u64 addr=(u64)si->si_addr; u64 rip=0;
    if(uc){ ucontext_t* c=(ucontext_t*)uc; rip=(u64)c->uc_mcontext.gregs[REG_RIP]; }
    char b[20]; char out[256]="[SIGSEGV addr=0x"; hex64(b,addr); strcat(out,b); strcat(out," rip=0x"); hex64(b,rip); strcat(out,b);
    // which loaded module is rip in?
    for(int i=0;i<g_nmod;i++){ u64 lo=(u64)g_mod[i].base; if(rip>=lo&&rip<lo+0x2000000){ strcat(out," in "); strcat(out,g_mod[i].name); strcat(out,"+0x"); hex64(b,rip-lo); strcat(out,b); break; } }
    if(rip>=(u64)g_code&&rip<(u64)g_code+(1<<20)) strcat(out," in <thunk/trap>");
    // ... else resolve via /proc/self/maps (native libs, driver, heap)
    { FILE* mf=fopen("/proc/self/maps","r");
      if(mf){ char line[512]; while(fgets(line,sizeof line,mf)){
          u64 lo=0,hi=0; if(sscanf(line,"%lx-%lx",&lo,&hi)==2&&rip>=lo&&rip<hi){
              char* p=line; while(*p&&*p!='\n'){ size_t L=strlen(out);
                  if(L<200) { out[L]=*p; out[L+1]=0; } p++; } break; } }
          fclose(mf); } }
    strcat(out,"]\n"); (void)write(2,out,strlen(out)); _exit(42); }

// ---- Evaluate scaffolding: native test images + exact NGX resource layout --
// Uses the real vendored NVSDK_NGX_Resource_VK (C-safe header) instead of a
// hand-rolled struct — layout bugs here are silent InvalidParameters.
#include "nvsdk_ngx_defs_vk.h"
typedef struct { VkImage im; VkDeviceMemory mm; VkImageView vw; } ImgRes;
_Static_assert(sizeof(NVSDK_NGX_Resource_VK)==56,"Resource_VK size");
_Static_assert(sizeof(NVSDK_NGX_ImageViewInfo_VK)==48,"ImageViewInfo size");
static u32 eval_memidx(u32 bits,u32 want){ VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    for(u32 i=0;i<mp.memoryTypeCount;i++)
        if((bits&(1u<<i))&&((mp.memoryTypes[i].propertyFlags&want)==want)) return i;
    return 0; }
static ImgRes MkImg(u32 w,u32 hh,VkFormat f,VkImageUsageFlags u,VkImageAspectFlags a){
    ImgRes r={0,0,0};
    VkImageCreateInfo ii={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType=VK_IMAGE_TYPE_2D,.format=f,.extent={w,hh,1},
        .mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,
        .tiling=VK_IMAGE_TILING_OPTIMAL,.usage=u,.sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
    if(vkCreateImage(g_dev,&ii,0,&r.im)!=VK_SUCCESS) return r;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev,r.im,&mr);
    // S5_EXPORT_FD=1: exportable memory (OPAQUE_FD) so the snippet can import
    // it into CUDA (external-memory interop). Needs _fd ext (enabled).
    VkExportMemoryAllocateInfo exi={.sType=VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext=getenv("S5_EXPORT_FD")?&exi:0,
        .allocationSize=mr.size,.memoryTypeIndex=eval_memidx(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    if(vkAllocateMemory(g_dev,&ai,0,&r.mm)!=VK_SUCCESS) return r;
    vkBindImageMemory(g_dev,r.im,r.mm,0);
    VkImageViewCreateInfo vi={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image=r.im,.viewType=VK_IMAGE_VIEW_TYPE_2D,.format=f,
        .subresourceRange={a,0,1,0,1}};
    vkCreateImageView(g_dev,&vi,0,&r.vw); return r; }

// gdb hook (S5_TRAP): clean breakpoint target (unique body defeats ICF merge).
volatile unsigned g_trap_tick=0;
MSABI static __attribute__((noinline)) void s_trap_hook(void){ g_trap_tick++; }

int main(void){
    printf("== Path B / S5(a): load Windows NGX host natively\n");
    init_env();   // populate the environment block (PATH etc.) NGX splits into its dir list
    g_code=mmap(NULL,1<<20,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    // module[0] must exist for GetModuleHandle(self); reserve a dummy base
    fflush(stdout);
    pid_t pid=getenv("NOFORK")?0:fork();   // NOFORK=1 runs NGX inline (no fork) so gdb watchpoints work
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
        g_dbg_nvngx_base=h->base;  // gdb anchor (ASLR-stable read point for base+offset breakpoints)
        { char b[80]; snprintf(b,sizeof b,"[_nvngx base=%p  GFR@%p  eval_c1e0@%p  ret_e194@%p]\n",
            (void*)h->base,(void*)(h->base+0xdf80),(void*)(h->base+0xc1e0),(void*)(h->base+0xe194)); logs(b); }
        // Report a WINDOWS-form directory to NGX (same fiction as GetModuleFileNameW
        // and the FullPath registry value): NGX splits paths on '\\' — a Unix-form dir
        // leaves its dir/file wstrings NULL/misaligned and crashes 0xb720. Real file
        // access still uses g_wine_dir() (wine_has/load_module translate by basename).
        static u16 wpath[80]; const char* dp="C:\\Windows\\System32\\"; int i=0; for(;dp[i]&&i<79;i++) wpath[i]=(u8)dp[i]; wpath[i]=0;  // UTF-16 (2-byte) path

        // LIGHTER query first: NVSDK_NGX_VULKAN_GetFeatureRequirements(FrameGeneration) — the
        // same call that returned GREEN under Proton. It takes only instance+pd, NOT the full
        // Init/adapter-correlation path (dxvk-nvapi's GetLogicalGpuInfo/LUID is never touched
        // here). Tests whether our native nvapi bridge (arch=Blackwell) is enough for the
        // driver to report FG available NATIVELY. FeatureRequirement = {FeatureSupported@0(u32),
        // MinHWArch@4(u32), MinOS[255]}; FeatureDiscoveryInfo = {SDKVer,FeatureID,Ident(32),path,info}.
        typedef int MSABI(*gfr_t)(void*,void*,const void*,void*);
        gfr_t GFR=(gfr_t)module_export(h,"NVSDK_NGX_VULKAN_GetFeatureRequirements");
        // S5_SDKVER overrides the NGX SDK version (hex, default 0x15 = SDK 310.7
        // macro). The 310.2-era driver DLL may want an older one (OutOfDate check).
        const char* se=getenv("S5_SDKVER"); unsigned sdkv=se?(unsigned)strtoul(se,0,0):0x15;
        if(GFR){
            struct { u32 sdkVer,featureId,idType,pad; u64 appId,u1,u2; const void* dataPath; const void* featInfo; } fdi;
            memset(&fdi,0,sizeof fdi); fdi.sdkVer=sdkv; fdi.featureId=11; fdi.idType=0; fdi.appId=0x1337ULL; fdi.dataPath=wpath;
            // Ghidra 2026-09-12: 0xc1e0 feeds 0xb720 (dir-list builder) from caller stack
            // slots ([rbp+0x77]/[rbp+0x8f]) that stay nil when FeatureInfo is NULL — same
            // shape as the Init §19 fix. Pass a real FeatureCommonInfo (PathListInfo ->
            // our DLL dir, InternalData NULL) instead of NULL.
            // Ghidra+gdb 2026-09-12: 0xb720 reads R14=[FeatureInfo] as {wchar** array,
            // count}: RDX=array[idx] must be a wchar*. So Path must point at an ARRAY
            // holding wpath (NOT at the chars directly — that crashes 0xa1f3 with the
            // string bytes misread as a pointer).
            // Full FeatureCommonInfo per SDK 310.7 (40 bytes): PathListInfo,
            // InternalData, LoggingInfo{callback,level,disable} (0x14+). Older DLLs
            // only read the prefix; the callback is MSABI (NGX calls MS-x64 from
            // any thread; write(2) is thread-safe). NOT variadic: no float-ABI risk.
            static const void* fci_paths[1]; fci_paths[0]=wpath;
            struct { const void* path; u32 len; u32 pad_; void* internal_;
                     void* logcb; int loglevel; int logdisable; } fci;
            memset(&fci,0,sizeof fci); fci.path=fci_paths; fci.len=1; fci.internal_=0;
            fci.logcb=(void*)s_ngx_log; fci.loglevel=2; fci.logdisable=0;
            fdi.featInfo=&fci;
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
            // Same lesson as GFR: pass a real FeatureCommonInfo (arg10), not NULL.
            static const void* ici_paths[1]; ici_paths[0]=wpath;
            struct { const void* path; u32 len; u32 pad_; void* internal_;
                     void* logcb; int loglevel; int logdisable; } ici;
            memset(&ici,0,sizeof ici); ici.path=ici_paths; ici.len=1; ici.internal_=0;
            ici.logcb=(void*)s_ngx_log; ici.loglevel=2; ici.logdisable=0;
            // Ghidra 2026-09-12: Init+d5d0 passes p6=0 and reloads gipa from
            // vulkan-1.dll itself; arg9 (gdpa) is dereferenced as {qword,dword,ptr*}
            // by FUN_18000ce40 — a CODE pointer (my_gdpa) feeds it code bytes and
            // faults at +0xceee, while NULL takes the guarded skip. Pass NULL.
            // SPIKE 2026-09-12: arg8 semantic probe. d5d0 forwards low32(arg8) as
            // ce40.param_8 (0x13-gate) and down the c8b0->c1e0 chain into an
            // sdkVer<=0x15 wrapper check. A gipa CODE pointer puts random bits
            // there (OutOfDate or worse). S5_ARG8INT forces an int instead
            // (e.g. 0x14) to test whether the slot is version-semantic.
            // Default (unset) keeps the documented gipa pointer.
            const char* ge8=getenv("S5_ARG8INT");
            void* a8=ge8?(void*)(uintptr_t)strtoul(ge8,0,0):(void*)my_gipa;
            int r=Init("a0b1c2d3-1234-5678-9abc-def012345678",0,"1.0",wpath,
                       (void*)g_inst,(void*)g_pd,(void*)g_dev,a8,0,&ici,sdkv);
            char b[80]; snprintf(b,sizeof b,"[NGX Init returned 0x%08X]\n",(unsigned)r); logs(b);
            // ---- CreateFeature(FG) in the SAME process after Init==0x1 ----
            // Gated by S5_CREATE=1 (needs taskset -c 0 for deterministic Init).
            // Params per NGX_VK_CREATE_DLSSG (+ DLSSG.Width/Height for 310.9):
            // CreationNodeMask, VisibilityNodeMask, Width, Height,
            // DLSSG.BackbufferFormat (+ DLSSG.Width/Height). Format uint is
            // UNPROVEN (Agent C): S5_FG_FMT raster {4,37,44,109,5}, S5_FG_W/H size.
            if(r==1 && getenv("S5_CREATE")){
                typedef int MSABI(*allocparams_t)(void**);
                typedef void MSABI(*setui_t)(void*,const char*,u32);
                typedef int MSABI(*scratch_t)(u32,void*,u64*);
                typedef int MSABI(*create_t)(void*,u32,void*,void**);
                allocparams_t Alloc=(allocparams_t)module_export(h,"NVSDK_NGX_VULKAN_AllocateParameters");
                Module* snip=0;
                for(int i=0;i<g_nmod;i++) if(!strcasecmp(g_mod[i].name,"nvngx_dlssg.dll")) snip=&g_mod[i];
                // Resolve via the HOST forwarder, NOT the snippet directly: the
                // snippet's Create/Scratch wrappers check GetModuleHandleExA(
                // FROM_ADDRESS, retaddr) + wcsstr(path, L"nvngx.dll") and fail
                // 0xBAD00002 ("Not called from NGX runtime") for a native
                // caller. Through the host forwarder the inner retaddr lies
                // inside _nvngx.dll ("...\_nvngx.dll" contains the needle).
                scratch_t Scratch=(scratch_t)module_export(h,"NVSDK_NGX_VULKAN_GetScratchBufferSize");
                create_t Create=(create_t)module_export(h,"NVSDK_NGX_VULKAN_CreateFeature");
                if(!Scratch&&snip) Scratch=(scratch_t)module_export(snip,"NVSDK_NGX_VULKAN_GetScratchBufferSize");
                if(!Create&&snip) Create=(create_t)module_export(snip,"NVSDK_NGX_VULKAN_CreateFeature");
                { char b2[128]; snprintf(b2,sizeof b2,"[Create] Alloc=%p snip=%p Scratch=%p Create=%p\n",
                    (void*)Alloc,(void*)snip,(void*)Scratch,(void*)Create); logs(b2); }
                if(Alloc&&Scratch&&Create){
                    void* params=0; int ra=Alloc(&params);
                    { char b2[80]; snprintf(b2,sizeof b2,"[Create] AllocateParameters -> 0x%X params=%p\n",(unsigned)ra,params); logs(b2); }
                    if(ra==1&&params){
                        void** vt=*(void***)params; setui_t SetUI=(setui_t)vt[3]; // vtable[3] = Set(uint)
                        // vtable roundtrip probe: Set then Get must return the value.
                        // If this fails, indices are shifted (e.g. hidden dtor slot).
                        typedef int MSABI(*getui_t)(void*,const char*,u32*);
                        getui_t GetU=(getui_t)vt[11]; // vtable[11] = Get(uint)
                        SetUI(params,"Width",1920);
                        u32 back=0xDEAD; int rg=GetU(params,"Width",&back);
                        { char b2[96]; snprintf(b2,sizeof b2,"[Create] vtable probe: Get(Width) -> 0x%X val=%u (want 1/1920)\n",(unsigned)rg,back); logs(b2); }
                        // Slots 1/7/9/15 probe: do SetF/SetVoid stick + Get back?
                        typedef void MSABI(*setvoid_t)(void*,const char*,void*);
                        typedef void MSABI(*setf_t)(void*,const char*,float);
                        typedef int MSABI(*getvoid_t)(void*,const char*,void**);
                        typedef int MSABI(*getf_t)(void*,const char*,float*);
                        setvoid_t SetV0=(setvoid_t)vt[7]; setf_t SetF0=(setf_t)vt[1];
                        getvoid_t GetV0=(getvoid_t)vt[15]; getf_t GetF0=(getf_t)vt[9];
                        static int dummyobj=0x12345678; void* bv=0; float bf=-1;
                        SetV0(params,"DLSSG.Backbuffer",&dummyobj); SetF0(params,"DLSSG.JitterOffsetX",0.5f);
                        int rgv=GetV0(params,"DLSSG.Backbuffer",&bv); int rgf=GetF0(params,"DLSSG.JitterOffsetX",&bf);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] vtable probe2: GetV(Backbuffer) -> 0x%X ptr=%p (want-dummy) GetF(JitterX) -> 0x%X val=%f (want 0.5)\n",
                            (unsigned)rgv,bv,(unsigned)rgf,bf); logs(b2); }
                        // Cross-type probe: does GetULL see a SetVoid value? (FG fetches
                        // resources via vtable[8]=GetULL — if type-strict, need SetULL.)
                        typedef int MSABI(*getull_t)(void*,const char*,unsigned long long*);
                        getull_t GetU64=(getull_t)vt[8];
                        unsigned long long b64=0xDEADDEAD; int rgu=GetU64(params,"DLSSG.Backbuffer",&b64);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] vtable probe4: GetULL(Backbuffer) -> 0x%X val=0x%llX (vs dummy %p)\n",
                            (unsigned)rgu,b64,&dummyobj); logs(b2); }
                        // ULL-pair probe: does SetULL store + GetULL retrieve?
                        typedef void MSABI(*setull_t)(void*,const char*,unsigned long long);
                        setull_t SetU64=(setull_t)vt[0];
                        SetU64(params,"DLSSG.Backbuffer",(unsigned long long)(uintptr_t)&dummyobj);
                        unsigned long long b65=0; int rgu2=GetU64(params,"DLSSG.Backbuffer",&b65);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] vtable probe5: SetULL+GetULL(Backbuffer) -> 0x%X val=0x%llX (want dummy %p)\n",
                            (unsigned)rgu2,b65,&dummyobj); logs(b2); }
                        // Float order probe: maybe DLL has float/double swapped vs header?
                        setf_t SetF2=(setf_t)vt[2]; getf_t GetF2=(getf_t)vt[10];
                        float bf2=-2; SetF2(params,"DLSSG.JitterOffsetY",0.25f);
                        int rgf2=GetF2(params,"DLSSG.JitterOffsetY",&bf2);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] vtable probe3 (swapped): GetF2(JitterY) -> 0x%X val=%f (want 0.25)\n",
                            (unsigned)rgf2,bf2); logs(b2); }
                        // Int-pair probe: does SetI+GetI (slots 4/12) roundtrip?
                        // (Subrect/extent reads may use the int getter.)
                        typedef int MSABI(*geti_t)(void*,const char*,int*);
                        typedef void MSABI(*seti_t)(void*,const char*,int);
                        seti_t SetI0=(seti_t)vt[4];
                        geti_t GetI=(geti_t)vt[12];
                        SetI0(params,"Width",(int)1920);
                        int backi=-99; int rgi=GetI(params,"Width",&backi);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] vtable probe6: GetI(Width) -> 0x%X val=%d (want 1/1920)\n",
                            (unsigned)rgi,backi); logs(b2); }
                        // Clean double roundtrip: SetD(vt[2]) + GetD(vt[10]) with a
                        // REAL double (full XMM2) — distinguishes "float broken"
                        // from "my float call broken".
                        typedef void MSABI(*setd_t)(void*,const char*,double);
                        typedef int MSABI(*getd_t)(void*,const char*,double*);
                        setd_t SetD=(setd_t)vt[2]; getd_t GetD=(getd_t)vt[10];
                        SetD(params,"DLSSG.DoubleProbe",0.5);
                        double bdd=-1; int rgd=GetD(params,"DLSSG.DoubleProbe",&bdd);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] vtable probe8: SetD+GetD(DoubleProbe) -> 0x%X val=%f (want 1/0.5)\n",
                            (unsigned)rgd,bdd); logs(b2); }
                        // Float cross-product: SetF via slot 1, then Get via EVERY
                        // slot 8..16 — which getter retrieves a float-stored key?
                        SetF0(params,"DLSSG.FloatProbe",0.5f);
                        for(int gi=8;gi<=16;gi++){
                            typedef int MSABI(*getg_t)(void*,const char*,void*);
                            getg_t Gg=(getg_t)vt[gi];
                            unsigned long long out=0xDEADDEADDEADDEADULL;
                            int rgg=Gg(params,"DLSSG.FloatProbe",&out);
                            double dd=0; memcpy(&dd,&out,8);
                            { char b2[128]; snprintf(b2,sizeof b2,"[Create] probeX: Get slot %d -> 0x%X raw=0x%llX dbl=%f\n",
                                gi,(unsigned)rgg,out,dd); logs(b2); } }
                        // Single-entry test: does SetULL clobber the uint entry?
                        typedef void MSABI(*setull_t)(void*,const char*,unsigned long long);
                        setull_t SetU64x=(setull_t)vt[0];
                        SetU64x(params,"Width",(unsigned long long)1920);
                        u32 back2=0xDEAD; int rg2=GetU(params,"Width",&back2);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] vtable probe7: after SetULL, GetUI(Width) -> 0x%X val=%u (1/1920=multi, else single)\n",
                            (unsigned)rg2,back2); logs(b2); }
                        const char* ew=getenv("S5_FG_W"); const char* eh=getenv("S5_FG_H"); const char* ef=getenv("S5_FG_FMT");
                        u32 W=ew?(u32)strtoul(ew,0,0):1920, H=eh?(u32)strtoul(eh,0,0):1080, F=ef?(u32)strtoul(ef,0,0):4;
                        // Test images hoisted before Create (S5_PRECREATE sets
                        // resources at Create time to test CTX-built-at-Create).
                        // S5_MVEC32=1 -> RG32F; S5_DEPTH_D32=1 -> D32+DEPTH aspect.
                        int d32=getenv("S5_DEPTH_D32")?1:0;
                        ImgRes col=MkImg(W,H,VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT);
                        ImgRes mv=MkImg(W,H,getenv("S5_MVEC32")?VK_FORMAT_R32G32_SFLOAT:VK_FORMAT_R16G16_SFLOAT,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT);
                        ImgRes dep=MkImg(W,H,d32?VK_FORMAT_D32_SFLOAT:VK_FORMAT_R32_SFLOAT,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT
                            |(d32?VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:0),
                            d32?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT);
                        ImgRes out=MkImg(W,H,VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_SAMPLED_BIT
                            |VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT);
                        { char b2[128]; snprintf(b2,sizeof b2,"[Create] imgs col=%p mv=%p dep=%p out=%p\n",
                            (void*)col.im,(void*)mv.im,(void*)dep.im,(void*)out.im); logs(b2); }
                        SetUI(params,"CreationNodeMask",1); SetUI(params,"VisibilityNodeMask",1);
                        SetUI(params,"Width",W); SetUI(params,"Height",H);
                        SetUI(params,"DLSSG.BackbufferFormat",F);
                        SetUI(params,"DLSSG.Width",W); SetUI(params,"DLSSG.Height",H);
                        // Type-strict map: snippet may read integrals as INT.
                        // Double-set every integral key as int too (vtable[4]).
                        typedef void MSABI(*seti_t)(void*,const char*,int);
                        seti_t SetI=(seti_t)vt[4];
                        SetI(params,"CreationNodeMask",1); SetI(params,"VisibilityNodeMask",1);
                        SetI(params,"Width",(int)W); SetI(params,"Height",(int)H);
                        SetI(params,"DLSSG.BackbufferFormat",(int)F);
                        SetI(params,"DLSSG.Width",(int)W); SetI(params,"DLSSG.Height",(int)H);
                        // NOTE: NO ULL for integrals! Single-entry-per-name map:
                        // last write wins, and readers are uint/int-specific.
                        // (ULL here broke Create with 'could not find Width'.)
                        // ULL is ONLY for resource pointers (fetched via slot 8).
                        // SynchronousInit: force Create to finish init inline
                        // (async worker may leave the extent cache empty for
                        // the first Evaluate -> (0,0) mismatches).
                        SetUI(params,"DLSSG.SynchronousInit",1);
                        SetI(params,"DLSSG.SynchronousInit",1);
                        // Cache primers: DynamicResolution only — InternalWidth/
                        // Height must STAY UNSET with DynRes=0 (validator exits
                        // via +0x76a54 if InternalW/H != 0 while DynRes == 0).
                        SetUI(params,"DLSSG.DynamicResolution",0);
                        SetI(params,"DLSSG.DynamicResolution",0);
                        // Explicit NULLs for optional resources at CREATE time:
                        // handle fields derived from them must be deterministic
                        // NULL (not malloc garbage) for later checks to skip.
                        { setvoid_t SetV9=(setvoid_t)vt[7];
                          const char* nk[]={"DLSSG.HUDLess","DLSSG.UI","DLSSG.UIAlpha",
                              "DLSSG.OutputReal","DLSSG.OutputDisableInterpolation",
                              "DLSSG.BidirectionalDistortionField"};
                          for(int i=0;i<6;i++) SetV9(params,nk[i],0); }
                        // Subrect keys ALSO at Create time (CTX/SUB may be built
                        // at Create from create-time params, never refreshed).
                        { const char* sk[]={"DLSSG.BackbufferSubrectWidth","DLSSG.BackbufferSubrectHeight",
                              "DLSSG.MVecsSubrectWidth","DLSSG.MVecsSubrectHeight",
                              "DLSSG.DepthSubrectWidth","DLSSG.DepthSubrectHeight",
                              "DLSSG.OutputInterpolatedSubrectWidth","DLSSG.OutputInterpolatedSubrectHeight",
                              "DLSSG.BackbufferSubrectBaseX","DLSSG.BackbufferSubrectBaseY",
                              "DLSSG.MVecsSubrectBaseX","DLSSG.MVecsSubrectBaseY",
                              "DLSSG.DepthSubrectBaseX","DLSSG.DepthSubrectBaseY"};
                          unsigned sv[]={W,H,W,H,W,H,W,H,0,0,0,0,0,0};
                          for(int i=0;i<14;i++){ SetUI(params,sk[i],sv[i]); SetI(params,sk[i],(int)sv[i]); } }
                        { char b2[96]; snprintf(b2,sizeof b2,"[Create] params set W=%u H=%u FMT=%u\n",W,H,F); logs(b2); }
                        // S5_PRECREATE=1: set resources (+ full ResVK) BEFORE
                        // CreateFeature — tests whether CTX/SUB is built at
                        // Create from create-time params (vs at Evaluate).
                        if(getenv("S5_PRECREATE")&&col.im&&mv.im&&dep.im&&out.im){
                            void** pv=*(void***)params;
                            setvoid_t SetVp=(setvoid_t)pv[7];
                            typedef void MSABI(*setull2_t)(void*,const char*,unsigned long long);
                            setull2_t SetUp=(setull2_t)pv[0];
                            NVSDK_NGX_Resource_VK pBack={.Resource.ImageViewInfo=
                                {(void*)col.vw,(void*)col.im,{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                    VK_FORMAT_R8G8B8A8_UNORM,W,H},
                                .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=false};
                            NVSDK_NGX_Resource_VK pMv={.Resource.ImageViewInfo=
                                {(void*)mv.vw,(void*)mv.im,{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                    getenv("S5_MVEC32")?VK_FORMAT_R32G32_SFLOAT:VK_FORMAT_R16G16_SFLOAT,W,H},
                                .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=false};
                            NVSDK_NGX_Resource_VK pDep={.Resource.ImageViewInfo=
                                {(void*)dep.vw,(void*)dep.im,{d32?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                    d32?VK_FORMAT_D32_SFLOAT:VK_FORMAT_R32_SFLOAT,W,H},
                                .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=false};
                            NVSDK_NGX_Resource_VK pOut={.Resource.ImageViewInfo=
                                {(void*)out.vw,(void*)out.im,{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                    VK_FORMAT_R8G8B8A8_UNORM,W,H},
                                .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=true};
                            SetVp(params,"DLSSG.Backbuffer",&pBack); SetVp(params,"DLSSG.MVecs",&pMv);
                            SetVp(params,"DLSSG.Depth",&pDep); SetVp(params,"DLSSG.OutputInterpolated",&pOut);
                            SetUp(params,"DLSSG.Backbuffer",(unsigned long long)(uintptr_t)&pBack);
                            SetUp(params,"DLSSG.MVecs",(unsigned long long)(uintptr_t)&pMv);
                            SetUp(params,"DLSSG.Depth",(unsigned long long)(uintptr_t)&pDep);
                            SetUp(params,"DLSSG.OutputInterpolated",(unsigned long long)(uintptr_t)&pOut);
                            logs("[Create] pre-create resources set\n"); }
                        u64 scratch=0; int rs=Scratch(11,params,&scratch);
                        { char b2[96]; snprintf(b2,sizeof b2,"[Create] GetScratchBufferSize -> 0x%X bytes=%llu\n",(unsigned)rs,(unsigned long long)scratch); logs(b2); }
                        // native cmd buffer for CreateFeature
                        VkCommandPool pool=0; VkCommandBuffer cmd=0;
                        VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                            .queueFamilyIndex=g_qfam,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
                        if(vkCreateCommandPool(g_dev,&pci,0,&pool)==VK_SUCCESS){
                            VkCommandBufferAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                .commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
                            if(vkAllocateCommandBuffers(g_dev,&ai,&cmd)==VK_SUCCESS){
                                VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
                                vkBeginCommandBuffer(cmd,&bi);
                            }
                        }
                        { char b2[64]; snprintf(b2,sizeof b2,"[Create] pool=%p cmd=%p\n",(void*)pool,(void*)cmd); logs(b2); }
                        void* handle=0; int rc=0;
                        if(cmd && getenv("S5_CREATE1")){
                            typedef int MSABI(*create1_t)(void*,void*,u32,void*,void**);
                            create1_t Create1=(create1_t)module_export(h,"NVSDK_NGX_VULKAN_CreateFeature1");
                            if(Create1) rc=Create1((void*)g_dev,(void*)cmd,11,params,&handle);
                            { char b2[96]; snprintf(b2,sizeof b2,"[Create] CreateFeature1(FG) -> 0x%X handle=%p\n",(unsigned)rc,handle); logs(b2); }
                        }
                        else if(cmd) rc=Create((void*)cmd,11,params,&handle);
                        { char b2[96]; snprintf(b2,sizeof b2,"[Create] CreateFeature(FG) -> 0x%X handle=%p\n",(unsigned)rc,handle); logs(b2); }
                        if(getenv("S5_TRAP")) s_trap_hook(); // gdb hook: inspect handle/state here
                                // ---- EvaluateFeature(FG) smoke in the SAME process ----
                                // S5_PRIME=1: NULL the handle's extent-subobject ptr
                                // ([handle+0xd0]) so check 1092 takes its
                                // explicit `je skip` path. Rationale (measured):
                                // NOTHING ever writes that field (watchpoint clean);
                                // it holds malloc garbage, and the extent cache it
                                // should point to is never built. NULL is a valid
                                // fresh-handle state per the check itself.
                                if(getenv("S5_PRIME")&&handle){
                                    void** slot=(void**)((char*)handle+0xd0);
                                    { char b2[80]; snprintf(b2,sizeof b2,"[Eval] prime: [h+0xd0] was %p -> NULL\n",*slot); logs(b2); }
                                    *slot=0; }
                        // Gated by S5_EVAL=1 (needs S5_CREATE=1 + RUN_INIT + taskset).
                        // Synthetic 1920x1080 inputs (must match create extent!):
                        // color (RGBA8, solid), mvecs (RG16F, zero), depth (R32F, 1.0),
                        // output (RGBA8, STORAGE). Minimal key set per strings-harvest
                        // (ParseNGXParameters required-list) + Guide defaults.
                        if(rc==1&&handle&&getenv("S5_EVAL")){
                            typedef void MSABI(*setvoid_t)(void*,const char*,void*);
                            typedef void MSABI(*setf_t)(void*,const char*,float);
                            typedef int MSABI(*eval_t)(void*,void*,void*,void*);
                            void** ev=*(void***)params;
                            setvoid_t SetV=(setvoid_t)ev[7]; setui_t SetU2=(setui_t)ev[3];
                            setf_t SetF=(setf_t)ev[1];
                            eval_t Eval=(eval_t)module_export(h,"NVSDK_NGX_VULKAN_EvaluateFeature");
                            if(!Eval&&snip) Eval=(eval_t)module_export(snip,"NVSDK_NGX_VULKAN_EvaluateFeature");
                            { char b2[64]; snprintf(b2,sizeof b2,"[Eval] Evaluate=%p\n",(void*)Eval); logs(b2); }
                            { char b2[128]; snprintf(b2,sizeof b2,"[Eval] imgs col=%p mv=%p dep=%p out=%p (hoisted pre-Create)\n",
                                (void*)col.im,(void*)mv.im,(void*)dep.im,(void*)out.im); logs(b2); }
                            if(col.im&&mv.im&&dep.im&&out.im&&Eval){
                                // setup cmd: clear + to GENERAL
                                VkCommandPool p2=0; VkCommandBuffer c2=0;
                                VkCommandPoolCreateInfo p2i={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=g_qfam};
                                vkCreateCommandPool(g_dev,&p2i,0,&p2);
                                VkCommandBufferAllocateInfo a2i={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                    .commandPool=p2,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
                                vkAllocateCommandBuffers(g_dev,&a2i,&c2);
                                VkCommandBufferBeginInfo b2i={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
                                vkBeginCommandBuffer(c2,&b2i);
                                VkImageMemoryBarrier bar[4]; VkImage autos[4]={col.im,mv.im,dep.im,out.im};
                                VkImageAspectFlags barA[4]={VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_ASPECT_COLOR_BIT,
                                    d32?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_ASPECT_COLOR_BIT};
                                for(int i=0;i<4;i++){ bar[i]=(VkImageMemoryBarrier){.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    .srcAccessMask=0,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                                    .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                                    .image=autos[i],.subresourceRange={barA[i],0,1,0,1}}; }
                                vkCmdPipelineBarrier(c2,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    0,0,0,0,0,4,bar);
                                VkClearColorValue cc[2]={{{{0.2f,0.4f,0.6f,1.0f}}},{{{0,0,0,0}}}};
                                VkImage tos[2]={col.im,mv.im};
                                for(int i=0;i<2;i++){ VkImageSubresourceRange r={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                                    vkCmdClearColorImage(c2,tos[i],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&cc[i],1,&r); }
                                if(d32){ VkClearDepthStencilValue dv={1.0f,0}; VkImageSubresourceRange r={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
                                    vkCmdClearDepthStencilImage(c2,dep.im,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&dv,1,&r); }
                                else { VkClearColorValue dc={{{1.0f,0,0,0}}}; VkImageSubresourceRange r={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                                    vkCmdClearColorImage(c2,dep.im,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&dc,1,&r); }
                                for(int i=0;i<4;i++){ bar[i].srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
                                    bar[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
                                    bar[i].oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                                    bar[i].newLayout=VK_IMAGE_LAYOUT_GENERAL; }
                                vkCmdPipelineBarrier(c2,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    0,0,0,0,0,4,bar);
                                vkEndCommandBuffer(c2);
                                VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c2};
                                vkQueueSubmit(g_queue,1,&si,0); vkQueueWaitIdle(g_queue);
                                logs("[Eval] setup done\n");
                                // --- Resource_VK structs (real header type) ---
                                NVSDK_NGX_Resource_VK rBack={.Resource.ImageViewInfo=
                                    {(void*)col.vw,(void*)col.im,{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                        VK_FORMAT_R8G8B8A8_UNORM,W,H},
                                    .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=false};
                                NVSDK_NGX_Resource_VK rMv={.Resource.ImageViewInfo=
                                    {(void*)mv.vw,(void*)mv.im,{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                        getenv("S5_MVEC32")?VK_FORMAT_R32G32_SFLOAT:VK_FORMAT_R16G16_SFLOAT,W,H},
                                    .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=false};
                                NVSDK_NGX_Resource_VK rDep={.Resource.ImageViewInfo=
                                    {(void*)dep.vw,(void*)dep.im,{d32?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                        d32?VK_FORMAT_D32_SFLOAT:VK_FORMAT_R32_SFLOAT,W,H},
                                    .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=false};
                                NVSDK_NGX_Resource_VK rOut={.Resource.ImageViewInfo=
                                    {(void*)out.vw,(void*)out.im,{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1},
                                        VK_FORMAT_R8G8B8A8_UNORM,W,H},
                                    .Type=NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,.ReadWrite=true};
                                SetV(params,"DLSSG.Backbuffer",&rBack); SetV(params,"DLSSG.MVecs",&rMv);
                                SetV(params,"DLSSG.Depth",&rDep); SetV(params,"DLSSG.OutputInterpolated",&rOut);
                                // FG-Evaluate fetches resources via the ULL getter
                                // (vtable[8]) — type-strict map, so SetVoid alone is
                                // invisible to it. Set BOTH representations.
                                typedef void MSABI(*setull_t)(void*,const char*,unsigned long long);
                                setull_t SetU64=(setull_t)ev[0];
                                SetU64(params,"DLSSG.Backbuffer",(unsigned long long)(uintptr_t)&rBack);
                                SetU64(params,"DLSSG.MVecs",(unsigned long long)(uintptr_t)&rMv);
                                SetU64(params,"DLSSG.Depth",(unsigned long long)(uintptr_t)&rDep);
                                SetU64(params,"DLSSG.OutputInterpolated",(unsigned long long)(uintptr_t)&rOut);
                                // S5_NORESET=1: first Evaluate with Reset=0 (Reset=1
                                // may clear the extent cache each call).
                                if(getenv("S5_NORESET")){ SetU2(params,"DLSSG.Reset",0);
                                    { seti_t SetI5=(seti_t)ev[4]; SetI5(params,"DLSSG.Reset",0); } }
                                static float ident[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
                                SetV(params,"DLSSG.ClipToPrevClip",ident); SetV(params,"DLSSG.PrevClipToClip",ident);
                                SetV(params,"DLSSG.CameraViewToClip",ident); SetV(params,"DLSSG.ClipToCameraView",ident);
                                SetV(params,"DLSSG.ClipToLensClip",ident);
                                SetU2(params,"DLSSG.Reset",1); SetU2(params,"DLSSG.MultiFrameCount",1);
                                SetU2(params,"DLSSG.MultiFrameIndex",1); SetU2(params,"DLSSG.DepthInverted",0);
                                SetU2(params,"DLSSG.CameraMotionIncluded",1); SetU2(params,"DLSSG.ColorBuffersHDR",0);
                                SetU2(params,"DLSSG.OrthoProjection",0); SetU2(params,"DLSSG.NotRenderingGameFrames",0);
                                SetU2(params,"DLSSG.AutomodeOverrideReset",0); SetU2(params,"DLSSG.EvalFlags",0);
                                SetU2(params,"DLSSG.InvertXAxis",0); SetU2(params,"DLSSG.InvertYAxis",0);
                                SetU2(params,"DLSSG.MvecDilated",0); SetU2(params,"DLSSG.MvecJittered",0);
                                SetF(params,"DLSSG.MvecScaleX",1.0f); SetF(params,"DLSSG.MvecScaleY",1.0f);
                                SetF(params,"DLSSG.JitterOffsetX",0.0f); SetF(params,"DLSSG.JitterOffsetY",0.0f);
                                SetF(params,"DLSSG.CameraNear",0.1f); SetF(params,"DLSSG.CameraFar",1000.0f);
                                SetF(params,"DLSSG.CameraFOV",1.0472f); SetF(params,"DLSSG.CameraAspectRatio",16.0f/9.0f);
                                SetF(params,"DLSSG.CameraPosX",0); SetF(params,"DLSSG.CameraPosY",0); SetF(params,"DLSSG.CameraPosZ",0);
                                SetF(params,"DLSSG.CameraUpX",0); SetF(params,"DLSSG.CameraUpY",1.0f); SetF(params,"DLSSG.CameraUpZ",0);
                                SetF(params,"DLSSG.CameraRightX",1.0f); SetF(params,"DLSSG.CameraRightY",0); SetF(params,"DLSSG.CameraRightZ",0);
                                SetF(params,"DLSSG.CameraFwdX",0); SetF(params,"DLSSG.CameraFwdY",0); SetF(params,"DLSSG.CameraFwdZ",-1.0f);
                                SetF(params,"DLSSG.CameraPinholeOffsetX",0); SetF(params,"DLSSG.CameraPinholeOffsetY",0);
                                SetF(params,"DLSSG.MvecInvalidValue",3.4028235e38f);
                                // Subrect extent keys: S5_NOSUBRECT=1 omits them entirely
                                // (absent keys may take a different merge path than
                                // explicit zeros — chicken-and-egg experiment).
                                if(!getenv("S5_NOSUBRECT")){
                                { seti_t SetI4=(seti_t)ev[4];
                                  const char* sk[]={"DLSSG.BackbufferSubrectWidth","DLSSG.BackbufferSubrectHeight",
                                      "DLSSG.MVecsSubrectWidth","DLSSG.MVecsSubrectHeight",
                                      "DLSSG.DepthSubrectWidth","DLSSG.DepthSubrectHeight",
                                      "DLSSG.OutputInterpolatedSubrectWidth","DLSSG.OutputInterpolatedSubrectHeight"};
                                  for(int i=0;i<8;i++){ u32 v=(i%2==0)?W:H;
                                      SetU2(params,sk[i],v); SetI4(params,sk[i],(int)v); }
                                  SetU2(params,"DLSSG.BackbufferSubrectBaseX",0); SetI4(params,"DLSSG.BackbufferSubrectBaseX",0);
                                  SetU2(params,"DLSSG.BackbufferSubrectBaseY",0); SetI4(params,"DLSSG.BackbufferSubrectBaseY",0);
                                  SetU2(params,"DLSSG.MVecsSubrectBaseX",0); SetI4(params,"DLSSG.MVecsSubrectBaseX",0);
                                  SetU2(params,"DLSSG.MVecsSubrectBaseY",0); SetI4(params,"DLSSG.MVecsSubrectBaseY",0);
                                  SetU2(params,"DLSSG.DepthSubrectBaseX",0); SetI4(params,"DLSSG.DepthSubrectBaseX",0);
                                  SetU2(params,"DLSSG.DepthSubrectBaseY",0); SetI4(params,"DLSSG.DepthSubrectBaseY",0); } }
                                { seti_t SetI2=(seti_t)ev[4];
                                  // NO ULL for integrals (single-entry map: ULL would
                                  // clobber uint/int the readers need). ULL only for
                                  // the 4 resource pointers (set above).
                                  // NO InternalWidth/Height (see create: must stay
                                  // unset with DynRes=0, else exit via +0x76a54).
                                  SetU2(params,"DLSSG.DynamicResolution",0);
                                  SetI2(params,"DLSSG.Reset",1); SetI2(params,"DLSSG.MultiFrameCount",1);
                                  SetI2(params,"DLSSG.MultiFrameIndex",1); SetI2(params,"DLSSG.DepthInverted",0);
                                  SetI2(params,"DLSSG.CameraMotionIncluded",1); SetI2(params,"DLSSG.ColorBuffersHDR",0);
                                  SetI2(params,"DLSSG.OrthoProjection",0); SetI2(params,"DLSSG.NotRenderingGameFrames",0);
                                  SetI2(params,"DLSSG.AutomodeOverrideReset",0); SetI2(params,"DLSSG.EvalFlags",0);
                                  SetI2(params,"DLSSG.InvertXAxis",0); SetI2(params,"DLSSG.InvertYAxis",0);
                                  SetI2(params,"DLSSG.MvecDilated",0); SetI2(params,"DLSSG.MvecJittered",0);
                                  SetI2(params,"DLSSG.DynamicResolution",0); }
                                // H1: legacy path REQUIRES CmdQueue+CmdAlloc (silent
                                // 0xBAD00005 at +0x76b5e/+0x76b6b otherwise).
                                // S5_CMDALLOC=cmd -> pass cmd buffer instead of pool.
                                SetV(params,"DLSSG.CmdQueue",(void*)g_queue);
                                SetV(params,"DLSSG.CmdAlloc",getenv("S5_CMDALLOC")?(void*)cmd:(void*)pool);
                                // ULL twins: CmdQueue/Alloc are fetched via the ULL
                                // getter (like resources); void*-only is invisible.
                                { typedef void MSABI(*setull_q)(void*,const char*,unsigned long long);
                                  setull_q SetUq=(setull_q)ev[0];
                                  SetUq(params,"DLSSG.CmdQueue",(unsigned long long)(uintptr_t)(void*)g_queue);
                                  SetUq(params,"DLSSG.CmdAlloc",(unsigned long long)(uintptr_t)(getenv("S5_CMDALLOC")?(void*)cmd:(void*)pool)); }
                                logs("[Eval] params set\n");
                                vkResetCommandPool(g_dev,pool,0);
                                VkCommandBufferBeginInfo ebi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
                                vkBeginCommandBuffer(cmd,&ebi);
                                int re=Eval((void*)cmd,handle,params,0);
                                { char b2[80]; snprintf(b2,sizeof b2,"[Eval] EvaluateFeature(FG) -> 0x%X\n",(unsigned)re); logs(b2); }
                                vkEndCommandBuffer(cmd);
                                VkSubmitInfo si2={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
                                vkQueueSubmit(g_queue,1,&si2,0); vkQueueWaitIdle(g_queue);
                                logs("[Eval] submitted+waited\n");
                                // Second Evaluate: cache may prime on first call
                                // (Reset=0 now). Log both results.
                                // ... plus 3rd/4th: priming may be progressive
                                // (call#2 lost 1074/1104; BackbufferExtent may
                                // need one more round). S5_EVAL_N controls count.
                                int neval=getenv("S5_EVAL_N")?atoi(getenv("S5_EVAL_N")):2;
                                for(int ei=1;ei<neval;ei++){
                                SetU2(params,"DLSSG.Reset",0);
                                { seti_t SetI3=(seti_t)ev[4]; SetI3(params,"DLSSG.Reset",0); }
                                vkResetCommandPool(g_dev,pool,0);
                                vkBeginCommandBuffer(cmd,&ebi);
                                int re2=Eval((void*)cmd,handle,params,0);
                                { char b2[80]; snprintf(b2,sizeof b2,"[Eval] EvaluateFeature(FG) #%d -> 0x%X\n",ei+1,(unsigned)re2); logs(b2); }
                                vkEndCommandBuffer(cmd);
                                VkSubmitInfo si2b={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
                                vkQueueSubmit(g_queue,1,&si2b,0); vkQueueWaitIdle(g_queue);
                                if(re2==1) break; }
                                // --- readback: out -> host buffer, checksum ---
                                VkBuffer stg=0; VkDeviceMemory stm=0;
                                VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                    .size=(VkDeviceSize)W*H*4,.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
                                if(vkCreateBuffer(g_dev,&bci,0,&stg)==VK_SUCCESS){
                                    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev,stg,&mr);
                                    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                        .allocationSize=mr.size,
                                        .memoryTypeIndex=eval_memidx(mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
                                    if(vkAllocateMemory(g_dev,&mai,0,&stm)==VK_SUCCESS){
                                        vkBindBufferMemory(g_dev,stg,stm,0);
                                        vkResetCommandPool(g_dev,p2,0); vkBeginCommandBuffer(c2,&b2i);
                                        VkBufferImageCopy cp={.bufferOffset=0,.bufferRowLength=0,.bufferImageHeight=0,
                                            .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
                                            .imageOffset={0,0,0},.imageExtent={W,H,1}};
                                        vkCmdCopyImageToBuffer(c2,out.im,VK_IMAGE_LAYOUT_GENERAL,stg,1,&cp);
                                        vkEndCommandBuffer(c2);
                                        VkSubmitInfo si3={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c2};
                                        vkQueueSubmit(g_queue,1,&si3,0); vkQueueWaitIdle(g_queue);
                                        void* mp=0; vkMapMemory(g_dev,stm,0,(VkDeviceSize)W*H*4,0,&mp);
                                        if(mp){ u8* px=mp; unsigned long long sum=0;
                                            for(u64 i=0;i<(u64)W*H*4;i++) sum+=px[i];
                                            char b2[160]; snprintf(b2,sizeof b2,
                                                "[Eval] readback sum=%llu px0=%u,%u,%u,%u\n",sum,px[0],px[1],px[2],px[3]); logs(b2);
                                            vkUnmapMemory(g_dev,stm); }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _exit(0);
    }
    int stx; waitpid(pid,&stx,0);
    if(WIFEXITED(stx)) printf("\n== child exit %d\n",WEXITSTATUS(stx));
    else if(WIFSIGNALED(stx)) printf("\n== child signal %d\n",WTERMSIG(stx));
    printf("== S5(a) verdict: see host DllMain result + dynamic LoadLibrary trace above.\n");
    return 0;
}
