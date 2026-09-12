// ms_thunk_typed.c — private Interop-Forschung (§69e UrhG), kein NVIDIA-Code.
// Nur eigene ABI-Stubs: per-Prototyp MS-x64 -> SysV Thunk-Generator nach
// libffi-Prinzip (Stub pro Signatur zur Laufzeit generiert, max 256 B).
//
// Kontext: s5_host.c/ms2sysv_common behandelt nur die INTEGER-Klasse
// (RCX/RDX/R8/R9 -> RDI/RSI/RDX/RCX). Diese Datei ergänzt GPR-Shuffle +
// XMM-Shuffle (MS-positional XMM0-3 -> SysV-separat XMM0..) + Stack-Nachbau
// (Args 5+), angedockt über make_ms2sysv_typed() (s. ANDOCK-Abschnitt unten).
//
// TODO (offen, NICHT implementiert):
//  - Struct-by-value-Split: SysV klassifiziert Structs in Eightbytes
//    (INTEGER/SSE/MEMORY), MS-x64 übergibt <=8 B im GPR bzw. sonst per Pointer.
//    Braucht pro-Signatur Eightbyte-Klassifizierung + Umpacken.
//  - Varargs-Spiegel: SysV-Variadics verlangen AL = Zahl der benutzten
//    Vektor-Register; dieser Stub setzt AL nicht (RAX trägt die Zieladresse).
//    Nur für feste (nicht-variadische) Prototypen verwenden.
//
// Bauen (NUR nach /tmp, Binary nie ins Repo):
//   gcc -O2 -o /tmp/ms_thunk_typed_test ms_thunk_typed.c && /tmp/ms_thunk_typed_test
#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

// ---- 1) Signatur-Beschreibung ----
typedef enum { K_INT, K_FLT, K_DBL, K_PTR } ArgK;
typedef struct { int nargs; ArgK arg[12]; int ret_is_float; } Sig;
// ret_is_float: nur dokumentarisch (1 = FP-Rückgabe, float ODER double).
// Rückgaben laufen in beiden ABIs identisch (RAX int / XMM0 float+double),
// daher kein Stub-Anteil.

// ---- Codegen (keine fremden Dependencies) ----
typedef unsigned char u8;
enum { R_RAX=0,R_RCX=1,R_RDX=2,R_RBX=3,R_RSP=4,R_RBP=5,R_RSI=6,R_RDI=7,
       R_R8=8,R_R9=9,R_R10=10,R_R11=11 };

static void e_mov64(u8**p,int dst,int src){          // mov dst,src (reg,reg)
    u8 rex=0x48; if(dst>=8)rex|=0x04; if(src>=8)rex|=0x01;
    *(*p)++=rex; *(*p)++=0x8B; *(*p)++=(u8)(0xC0|((dst&7)<<3)|(src&7));
}
static void e_mov_mem_rsp(u8**p,int dst,int disp){    // mov dst,[rsp+disp]
    u8 rex=0x48; if(dst>=8)rex|=0x04;
    *(*p)++=rex; *(*p)++=0x8B;
    if(disp<128){ *(*p)++=(u8)(0x44|((dst&7)<<3)); *(*p)++=0x24; *(*p)++=(u8)disp; }
    else { *(*p)++=(u8)(0x84|((dst&7)<<3)); *(*p)++=0x24;
        *(*p)++=(u8)disp; *(*p)++=(u8)(disp>>8);
        *(*p)++=(u8)(disp>>16); *(*p)++=(u8)(disp>>24); }
}
static void e_movaps(u8**p,int dst,int src){          // movaps xmm_dst,xmm_src
    *(*p)++=0x0F; *(*p)++=0x28; *(*p)++=(u8)(0xC0|((dst&7)<<3)|(src&7));
}
static void e_push_mem_rsp(u8**p,int disp){           // push qword [rsp+disp]
    if(disp<128){ *(*p)++=0xFF; *(*p)++=0x74; *(*p)++=0x24; *(*p)++=(u8)disp; }
    else { *(*p)++=0xFF; *(*p)++=0xB4; *(*p)++=0x24;
        *(*p)++=(u8)disp; *(*p)++=(u8)(disp>>8);
        *(*p)++=(u8)(disp>>16); *(*p)++=(u8)(disp>>24); }
}
static void e_sub_rsp(u8**p,int n){
    *(*p)++=0x48; *(*p)++=0x83; *(*p)++=0xEC; *(*p)++=(u8)n; }
static void e_add_rsp(u8**p,int n){
    *(*p)++=0x48; *(*p)++=0x83; *(*p)++=0xC4; *(*p)++=(u8)n; }
static void e_movabs_rax(u8**p,void* v){              // movabs rax,imm64
    *(*p)++=0x48; *(*p)++=0xB8; memcpy(*p,&v,8); *p+=8; }
static void e_call_rax(u8**p){ *(*p)++=0xFF; *(*p)++=0xD0; }
static void e_ret(u8**p){ *(*p)++=0xC3; }

// Kern: emittiert den Stub nach *buf, gibt Länge zurück (0 = Fehler).
// Layout bei Eintritt (MS-x64): RCX/RDX/R8/R9 + XMM0-3 (positional),
// [RSP+0x28+(i-4)*8] = Stack-Args 5+.
size_t emit_ms2sysv_typed(u8* buf, void* target, const Sig* s){
    if(!buf || !target || !s) return 0;
    if(s->nargs < 0 || s->nargs > 12) return 0;
    for(int i=0;i<s->nargs;i++)
        if(s->arg[i]!=K_INT && s->arg[i]!=K_FLT &&
           s->arg[i]!=K_DBL && s->arg[i]!=K_PTR) return 0;
    (void)s->ret_is_float; // kein Code-Effekt (s. Kommentar bei Sig)
    u8* p = buf;
    static const int MS_GPR[4] = { R_RCX, R_RDX, R_R8, R_R9 };
    static const int SV_GPR[6] = { R_RDI, R_RSI, R_RDX, R_RCX, R_R8, R_R9 };
    int ipos[12], nint=0, spos[12], nsse=0;
    for(int i=0;i<s->nargs;i++){
        if(s->arg[i]==K_INT || s->arg[i]==K_PTR) ipos[nint++]=i;
        else spos[nsse++]=i;
    }
    // 1) GPR-Shuffle: MS-GPRs sichern (R9 bleibt live, wird zuerst verbraucht:
    //    eine MS-R9-Quelle landet stets in SysV-Slot 0..3, nie in R8/R9/Stack).
    e_mov64(&p,R_R10,R_RCX);
    e_mov64(&p,R_R11,R_RDX);
    e_mov64(&p,R_RAX,R_R8);
    int nreg = nint<6 ? nint : 6;
    for(int j=0;j<nreg;j++){
        int i=ipos[j], dst=SV_GPR[j];
        if(i<4){
            int ms=MS_GPR[i];
            int saved = (ms==R_RCX)?R_R10:(ms==R_RDX)?R_R11:(ms==R_R8)?R_RAX:R_R9;
            e_mov64(&p,dst,saved);
        }else{
            e_mov_mem_rsp(&p,dst,0x28+(i-4)*8); // vor sub/push: Eintritts-Displacement
        }
    }
    // 2) XMM-Shuffle: MS-positional -> SysV-separat, aufsteigend
    //    (dst<=src stets, daher clobber-frei). movaps kopiert bit-exakt.
    int nxmm = nsse<8 ? nsse : 8;
    for(int j=0;j<nxmm;j++){ int src=spos[j]; if(src!=j) e_movaps(&p,j,src); }
    // 3) SysV-Stack: Überlauf-Args (Int-Rang>=6, SSE-Rang>=8) sind stets
    //    MS-Stack-Slots (i>=4). Per push in umgekehrter Reihenfolge, 16B-Align:
    //    Eintritt RSP%16==8 -> vor call muss RSP%16==0 gelten.
    int stk[12], nstk=0, ri=0, rs=0;
    for(int i=0;i<s->nargs;i++){
        if(s->arg[i]==K_INT || s->arg[i]==K_PTR){ if(ri>=6) stk[nstk++]=i; ri++; }
        else { if(rs>=8) stk[nstk++]=i; rs++; }
    }
    int pad = (nstk%2==0) ? 8 : 0;
    if(pad) e_sub_rsp(&p,pad);
    for(int k=0;k<nstk;k++){
        int i=stk[nstk-1-k]; // letzter Stack-Arg zuerst (höchste Adresse)
        e_push_mem_rsp(&p, 0x28+(i-4)*8+pad+8*k); // +pad+8k: RSP wanderte
    }
    // 4) Ziel aufrufen (call, nicht jmp: Stack-Cleanup danach), aufräumen, ret.
    e_movabs_rax(&p,target); // RAX-Temp ist hier frei (MS-R8 längst verbraucht)
    e_call_rax(&p);
    e_add_rsp(&p,nstk*8+pad);
    e_ret(&p);
    return (size_t)(p-buf);
}

// ---- 2) ms_typed: ausführbarer Stub via mmap (RWX, max 256 B) ----
void* ms_typed(void* target, const Sig* s){
    u8* mem = mmap(0, 4096, PROT_READ|PROT_WRITE|PROT_EXEC,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(mem == MAP_FAILED) return 0;
    size_t n = emit_ms2sysv_typed(mem, target, s);
    if(n == 0 || n > 256){ return 0; } // 256B-Limit (leck bei Fehler egal: Test)
    return mem;
}

// ---- ANDOCK an s5_host.c: make_ms2sysv_typed (drop-in neben make_ms2sysv) ----
// Signatur: void* make_ms2sysv_typed(void* target, const Sig* s);
// s5_host.c kann alternativ emit_ms2sysv_typed() direkt in seine g_code-Arena
// schreiben (g_codeoff += emit_ms2sysv_typed(g_code+g_codeoff, target, sig))
// statt diese statische Arena zu nutzen.
static u8 g_typed_arena[8192];
static size_t g_typed_off = 0;
void* make_ms2sysv_typed(void* target, const Sig* s){
    size_t n = emit_ms2sysv_typed(g_typed_arena+g_typed_off, target, s);
    if(n == 0 || n > 256 || g_typed_off+n > sizeof g_typed_arena) return 0;
    void* st = g_typed_arena+g_typed_off;
    g_typed_off += n;
    return st;
}

// ---- 3) Pilot-Signaturen + Selbsttest (Technik aus float_abi_spike.c) ----
#define MSABI __attribute__((ms_abi))
static uint64_t dbits(double d){ uint64_t u; memcpy(&u,&d,8); return u; }
static uint32_t fbits(float f){ uint32_t u; memcpy(&u,&f,4); return u; }

// (a) cuLaunchKernel-ähnlich: 11x INT/PTR (STACK-Pilot, 5 SysV-Stack-Args)
static uintptr_t got11[11];
static long stk11_impl(uintptr_t a0,uintptr_t a1,uintptr_t a2,uintptr_t a3,
        uintptr_t a4,uintptr_t a5,uintptr_t a6,uintptr_t a7,
        uintptr_t a8,uintptr_t a9,uintptr_t a10){
    got11[0]=a0; got11[1]=a1; got11[2]=a2; got11[3]=a3; got11[4]=a4;
    got11[5]=a5; got11[6]=a6; got11[7]=a7; got11[8]=a8; got11[9]=a9; got11[10]=a10;
    uintptr_t t=a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10; return (long)t;
}
MSABI static long ms_stk11(uintptr_t a0,uintptr_t a1,uintptr_t a2,uintptr_t a3,
        uintptr_t a4,uintptr_t a5,uintptr_t a6,uintptr_t a7,
        uintptr_t a8,uintptr_t a9,uintptr_t a10){
    return stk11_impl(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10);
}
typedef long MSABI(*stk11_ms_t)(uintptr_t,uintptr_t,uintptr_t,uintptr_t,
        uintptr_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t);

// (b) vkCmdSetDepthBounds-ähnlich: (PTR,FLOAT,FLOAT)
static void* gotDbC; static float gotMn, gotMx;
static float db_impl(void* c, float mn, float mx){
    gotDbC=c; gotMn=mn; gotMx=mx; return mn+mx;
}
MSABI static float ms_db(void* c, float mn, float mx){ return db_impl(c,mn,mx); }
typedef float MSABI(*db_ms_t)(void*,float,float);

// (c) gemischt: (INT,FLOAT,INT,DOUBLE + 3 Stack-INTs)
static long gotM_a,gotM_c,gotM_e,gotM_f,gotM_g; static float gotM_b; static double gotM_d;
static double mix_impl(long a,float b,long c,double d,long e,long f,long g){
    gotM_a=a; gotM_b=b; gotM_c=c; gotM_d=d; gotM_e=e; gotM_f=f; gotM_g=g;
    return (double)a+(double)b+(double)c+d+(double)e+(double)f+(double)g;
}
MSABI static double ms_mix(long a,float b,long c,double d,long e,long f,long g){
    return mix_impl(a,b,c,d,e,f,g);
}
typedef double MSABI(*mix_ms_t)(long,float,long,double,long,long,long);

static int fails = 0;
#define CHECK(cond, ...) do { \
    if(cond){ printf("  ok: " __VA_ARGS__); printf("\n"); } \
    else { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } \
} while(0)

int main(void){
    printf("[a] cuLaunchKernel-ähnlich: 11x INT/PTR (STACK-Pilot)\n");
    static const Sig sig11 = { 11,
        { K_PTR,K_INT,K_INT,K_INT,K_INT,K_INT,K_INT,K_INT,K_INT,K_INT,K_PTR }, 0 };
    uintptr_t v11[11] = { (uintptr_t)&v11,
        0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL,
        0x4444444444444444ULL, 0x5555555555555555ULL, 0x6666666666666666ULL,
        0x7777777777777777ULL, 0x8888888888888888ULL, 0x9999999999999999ULL,
        (uintptr_t)&fails };
    stk11_ms_t t11 = (stk11_ms_t)ms_typed((void*)stk11_impl, &sig11);
    assert(t11 != 0);
    { u8 tmp[256]; size_t n = emit_ms2sysv_typed(tmp,(void*)stk11_impl,&sig11);
      printf("  stub11: %zu B\n", n); assert(n > 0 && n <= 256); }
    long r_ref = ms_stk11(v11[0],v11[1],v11[2],v11[3],v11[4],v11[5],
                           v11[6],v11[7],v11[8],v11[9],v11[10]);
    uintptr_t snap[11]; memcpy(snap,got11,sizeof snap);
    memset(got11,0,sizeof got11);
    long r_new = t11(v11[0],v11[1],v11[2],v11[3],v11[4],v11[5],
                     v11[6],v11[7],v11[8],v11[9],v11[10]);
    for(int i=0;i<11;i++) CHECK(got11[i]==snap[i] && got11[i]==v11[i], "arg%d bit-exakt", i);
    CHECK(r_new==r_ref, "return bit-exakt");

    printf("[b] vkCmdSetDepthBounds-ähnlich: (PTR,FLOAT,FLOAT)\n");
    static const Sig sigDb = { 3, { K_PTR, K_FLT, K_FLT }, 1 };
    void* c = (void*)&fails; float mn = -0.5f, mx = 3.14159f;
    db_ms_t tdb = (db_ms_t)ms_typed((void*)db_impl, &sigDb);
    assert(tdb != 0);
    { u8 tmp[256]; size_t n = emit_ms2sysv_typed(tmp,(void*)db_impl,&sigDb);
      printf("  stubDb: %zu B\n", n); assert(n > 0 && n <= 256); }
    float rb_ref = ms_db(c,mn,mx);
    void* sC=gotDbC; float sMn=gotMn, sMx=gotMx;
    gotDbC=0; gotMn=0; gotMx=0;
    float rb_new = tdb(c,mn,mx);
    CHECK(gotDbC==sC && gotDbC==c, "ptr bit-exakt");
    CHECK(fbits(gotMn)==fbits(sMn) && fbits(gotMn)==fbits(mn), "mn bit-exakt (XMM1->XMM0)");
    CHECK(fbits(gotMx)==fbits(sMx) && fbits(gotMx)==fbits(mx), "mx bit-exakt (XMM2->XMM1)");
    CHECK(fbits(rb_new)==fbits(rb_ref), "return bit-exakt");

    printf("[c] gemischt: (INT,FLOAT,INT,DOUBLE + 3 Stack-INTs)\n");
    static const Sig sigMx = { 7,
        { K_INT,K_FLT,K_INT,K_DBL,K_INT,K_INT,K_INT }, 1 };
    long a=-123456789012345LL, cc=0x23456789ABCDEFLL,
         e=55, f=66, g=77; float b=0.125f; double d=3.141592653589793;
    mix_ms_t tmx = (mix_ms_t)ms_typed((void*)mix_impl, &sigMx);
    assert(tmx != 0);
    { u8 tmp[256]; size_t n = emit_ms2sysv_typed(tmp,(void*)mix_impl,&sigMx);
      printf("  stubMx: %zu B\n", n); assert(n > 0 && n <= 256); }
    double rm_ref = ms_mix(a,b,cc,d,e,f,g);
    long sA=gotM_a,sCc=gotM_c,sE=gotM_e,sF=gotM_f,sG=gotM_g;
    float sB=gotM_b; double sD=gotM_d;
    gotM_a=gotM_c=gotM_e=gotM_f=gotM_g=0; gotM_b=0; gotM_d=0;
    double rm_new = tmx(a,b,cc,d,e,f,g);
    CHECK(gotM_a==sA && gotM_a==a, "a bit-exakt (RCX->RDI)");
    CHECK(fbits(gotM_b)==fbits(sB) && fbits(gotM_b)==fbits(b), "b bit-exakt (XMM1->XMM0)");
    CHECK(gotM_c==sCc && gotM_c==cc, "c bit-exakt (R8->RSI)");
    CHECK(dbits(gotM_d)==dbits(sD) && dbits(gotM_d)==dbits(d), "d bit-exakt (XMM3->XMM1)");
    CHECK(gotM_e==sE && gotM_e==e, "e bit-exakt (Stack->RDX)");
    CHECK(gotM_f==sF && gotM_f==f, "f bit-exakt (Stack->RCX)");
    CHECK(gotM_g==sG && gotM_g==g, "g bit-exakt (Stack->R8)");
    CHECK(dbits(rm_new)==dbits(rm_ref), "return bit-exakt");

    assert(fails == 0);
    if(fails==0){ printf("PASS: alle typisierten Thunks bit-exakt\n"); return 0; }
    printf("FAIL: %d Checks fehlgeschlagen\n", fails);
    return 1;
}
