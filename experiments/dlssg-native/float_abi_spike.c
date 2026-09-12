// float_abi_spike.c — private Interop-Forschung (§69e UrhG).
// Kein NVIDIA-Code, keine DLL-Auszüge: nur eigene ABI-Testfunktionen.
// Zweck: beweist, dass MS-x64 -> SysV Calls mit float/double + struct-by-value
// bit-exakt übergeben werden (Vorarbeit für CUDA-Interop mit floats in
// EvaluateFeature). Eigenständig: gcc -o /tmp/float_abi_spike float_abi_spike.c
// Binary gehört nach /tmp, nie ins Repo.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MSABI __attribute__((ms_abi))

static uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint32_t fbits(float f)  { uint32_t u; memcpy(&u, &f, 4); return u; }

// ---- SysV-Zielfunktionen (nativ, eigene Testfunktionen) ----
double spike_f_impl(double a, int b, double c);
float  spike_g_impl(int a, float b, int c, double d);
typedef struct { double x; int y; } spike_S; // 16-Byte struct by value
_Static_assert(sizeof(spike_S) == 16, "spike_S muss 16 Bytes haben");
double spike_h_impl(spike_S s, int n);
double spike_stk_impl(int a, int b, int c, int d, int e, int f, int g, double h);

// Empfangs-Logs der Callees (bit-exakt geprüft)
static double got_f_a, got_f_c; static int got_f_b;
static int got_g_a, got_g_c; static float got_g_b; static double got_g_d;
static spike_S got_h_s; static int got_h_n;
static int got_stk_a, got_stk_b, got_stk_c, got_stk_d,
           got_stk_e, got_stk_f, got_stk_g; static double got_stk_h;

double spike_f_impl(double a, int b, double c) {
    got_f_a = a; got_f_b = b; got_f_c = c;
    return a + (double)b + c;
}
float spike_g_impl(int a, float b, int c, double d) {
    got_g_a = a; got_g_b = b; got_g_c = c; got_g_d = d;
    return (float)((double)a + (double)b + (double)c + d);
}
double spike_h_impl(spike_S s, int n) {
    got_h_s = s; got_h_n = n;
    return s.x + (double)s.y + (double)n;
}
double spike_stk_impl(int a, int b, int c, int d, int e, int f, int g, double h) {
    got_stk_a = a; got_stk_b = b; got_stk_c = c; got_stk_d = d;
    got_stk_e = e; got_stk_f = f; got_stk_g = g; got_stk_h = h;
    return (double)a + b + c + d + e + f + g + h;
}

// ---- Referenzpfad: Compiler-generiert (GCC kennt beide ABIs) ----
// main (SysV) ruft diese ms_abi-Wrapper auf -> Aufruf erfolgt MS-x64
// (RCX/RDX/R8/R9 + XMM0-3 + Shadow + Stack). Der Wrapper liest die
// MS-Argumente aus und der Compiler stellt sie dem SysV-Callee korrekt zu.
MSABI static double ms_f(double a, int b, double c) { return spike_f_impl(a, b, c); }
MSABI static float  ms_g(int a, float b, int c, double d) { return spike_g_impl(a, b, c, d); }
MSABI static double ms_h(spike_S s, int n) { return spike_h_impl(s, n); }
MSABI static double ms_stk(int a, int b, int c, int d, int e, int f, int g, double h) {
    return spike_stk_impl(a, b, c, d, e, f, g, h);
}

// ---- Manueller Pfad: float-aware Thunks (nasm-frei, Top-Level inline-asm) ----
// Gleiche Einstiegssituation wie ms2sysv_common in s5_host.c (MS-x64-Layout),
// aber mit explizitem XMM-Shuffle + Stack-Nachbau. Pro Signatur ein Stub,
// weil das XMM0-3(MS, positional) -> XMM0-7(SysV, SSE-separat) Mapping sowie
// die Struct-Klassifizierung signaturabhängig sind (generischer Laufzeit-Thunk
// bräuchte pro Prototyp generierte Stubs, vgl. libffi-Ansatz).
//   f: MS(XMM0=a, RDX=b, XMM2=c)            -> SysV(XMM0=a, RDI=b, XMM1=c)
//   g: MS(RCX=a, XMM1=b, R8=c, XMM3=d)      -> SysV(RDI=a, XMM0=b, RSI=c, XMM1=d)
//   h: MS(RCX=&s [16B-Struct per Pointer], RDX=n)
//                                              -> SysV(XMM0=s.x, RDI=s.y, RSI=n)
//   stk: MS(RCX..R9=a..d, Stack=e,f,g,h)    -> SysV(RDI..R9=a..f, Stack=g, XMM0=h)
MSABI double manual_f(double a, int b, double c);
MSABI float  manual_g(int a, float b, int c, double d);
MSABI double manual_h(spike_S s, int n);
// stk braucht Stack-Nachbau (Shadow entfernen): eigener Stub s. unten.
MSABI double manual_stk_fixed(int a, int b, int c, int d, int e, int f, int g, double h);

__asm__(
".text\n"
".globl manual_f\n"
"manual_f:\n"
"  mov %rdx, %rdi\n"          // SysV arg2 (int) = MS arg2
"  movapd %xmm2, %xmm1\n"     // SysV XMM1 (2. SSE) = MS XMM2 (Pos. 2)
"  jmp spike_f_impl\n"        // XMM0 bleibt a; RSP-Lage passt (reiner jmp)\n"
".globl manual_g\n"
"manual_g:\n"
"  mov %rcx, %rdi\n"          // SysV RDI (1. int) = MS RCX (Pos. 0)
"  mov %r8, %rsi\n"           // SysV RSI (2. int) = MS R8 (Pos. 2)
"  movaps %xmm1, %xmm0\n"     // SysV XMM0 (1. float) = MS XMM1 (Pos. 1)
"  movaps %xmm3, %xmm1\n"     // SysV XMM1 (2. float) = MS XMM3 (Pos. 3)
"  jmp spike_g_impl\n"
".globl manual_h\n"
"manual_h:\n"
"  movsd (%rcx), %xmm0\n"     // SysV XMM0 = s.x (SSE-Eightbyte des Structs)
"  mov 8(%rcx), %edi\n"       // SysV RDI = s.y (INTEGER-Eightbyte, zero-extended)
"  mov %rdx, %rsi\n"          // SysV RSI = n (MS Pos. 1)
"  jmp spike_h_impl\n"
// stk: R10/R11 sind in beiden ABIs volatil und dienen als Rangierregister.
// MS-Layout bei Eintritt: RCX..R9 = a..d, [RSP+8..+40) Shadow,
// [RSP+0x28]=e, [RSP+0x30]=f, [RSP+0x38]=g, [RSP+0x40]=h (double).
".globl manual_stk_fixed\n"
"manual_stk_fixed:\n"
"  mov %rcx, %rdi\n"          // SysV RDI (a) = MS RCX — zuerst, vor RCX-Clobber
"  mov %rdx, %rsi\n"          // SysV RSI (b) = MS RDX
"  mov %r8, %rdx\n"           // SysV RDX (c) = MS R8
"  mov %r9, %rcx\n"           // SysV RCX (d) = MS R9
"  mov 0x28(%rsp), %r8\n"     // SysV R8 (e) = MS Stack-Slot 4
"  mov 0x30(%rsp), %r9\n"     // SysV R9 (f) = MS Stack-Slot 5
"  movsd 0x40(%rsp), %xmm0\n" // SysV XMM0 (h) = MS Stack-Slot 7 (double)
"  mov 0x38(%rsp), %rax\n"    // SysV Stack-Slot (g) = MS Stack-Slot 6
"  push %rax\n"               // SysV Stack-Arg aufbauen (alignt: Eintritt %16==8)
"  call spike_stk_impl\n"
"  add $8, %rsp\n"
"  ret\n"
);
MSABI double manual_stk_fixed(int a, int b, int c, int d, int e, int f, int g, double h);

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (cond) { printf("  ok: " __VA_ARGS__); printf("\n"); } \
    else { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } \
} while (0)

int main(void) {
    // Testwerte mit interessanten Bitmustern
    double fa = 1.5, fc = -2.75; int fb = 0x12345678;
    int ga = -7; float gb = 0.125f; int gc = 0x23456789; double gd = 3.141592653589793;
    spike_S hs = { 2.5, 42 }; int hn = -99;
    int sa = 11, sb = 22, sc = 33, sd = 44, se = 55, sf = 66, sg = 77;
    double sh = 9.87654321;

    double exp_f = fa + (double)fb + fc;
    float  exp_g = (float)((double)ga + (double)gb + (double)gc + gd);
    double exp_h = hs.x + (double)hs.y + (double)hn;
    double exp_stk = (double)sa + sb + sc + sd + se + sf + sg + sh;

    printf("[1] Referenz ms_abi->SysV: f(double,int,double)\n");
    { double r = ms_f(fa, fb, fc);
      CHECK(dbits(got_f_a) == dbits(fa), "a bit-exakt");
      CHECK(got_f_b == fb, "b bit-exakt");
      CHECK(dbits(got_f_c) == dbits(fc), "c bit-exakt");
      CHECK(dbits(r) == dbits(exp_f), "return bit-exakt"); }

    printf("[2] Referenz ms_abi->SysV: g(int,float,int,double)\n");
    { float r = ms_g(ga, gb, gc, gd);
      CHECK(got_g_a == ga, "a bit-exakt");
      CHECK(fbits(got_g_b) == fbits(gb), "b bit-exakt");
      CHECK(got_g_c == gc, "c bit-exakt");
      CHECK(dbits(got_g_d) == dbits(gd), "d bit-exakt");
      CHECK(fbits(r) == fbits(exp_g), "return bit-exakt"); }

    printf("[3] Referenz ms_abi->SysV: h(16B-struct,int)\n");
    { double r = ms_h(hs, hn);
      CHECK(dbits(got_h_s.x) == dbits(hs.x), "s.x bit-exakt");
      CHECK(got_h_s.y == hs.y, "s.y bit-exakt");
      CHECK(got_h_n == hn, "n bit-exakt");
      CHECK(dbits(r) == dbits(exp_h), "return bit-exakt"); }

    printf("[4] Referenz ms_abi->SysV: stk(7xint,double, Stack-Fall)\n");
    { double r = ms_stk(sa, sb, sc, sd, se, sf, sg, sh);
      CHECK(got_stk_a == sa && got_stk_b == sb && got_stk_c == sc && got_stk_d == sd,
            "a..d bit-exakt");
      CHECK(got_stk_e == se && got_stk_f == sf && got_stk_g == sg, "e..g bit-exakt");
      CHECK(dbits(got_stk_h) == dbits(sh), "h bit-exakt");
      CHECK(dbits(r) == dbits(exp_stk), "return bit-exakt"); }

    printf("[5] Manuell float-aware: f\n");
    { double r = manual_f(fa, fb, fc);
      CHECK(dbits(got_f_a) == dbits(fa), "a bit-exakt (XMM0 erhalten)");
      CHECK(got_f_b == fb, "b bit-exakt (RDX->RDI)");
      CHECK(dbits(got_f_c) == dbits(fc), "c bit-exakt (XMM2->XMM1)");
      CHECK(dbits(r) == dbits(exp_f), "return bit-exakt"); }

    printf("[6] Manuell float-aware: g\n");
    { float r = manual_g(ga, gb, gc, gd);
      CHECK(got_g_a == ga, "a bit-exakt (RCX->RDI)");
      CHECK(fbits(got_g_b) == fbits(gb), "b bit-exakt (XMM1->XMM0)");
      CHECK(got_g_c == gc, "c bit-exakt (R8->RSI)");
      CHECK(dbits(got_g_d) == dbits(gd), "d bit-exakt (XMM3->XMM1)");
      CHECK(fbits(r) == fbits(exp_g), "return bit-exakt"); }

    printf("[7] Manuell float-aware: h (MS-Pointer -> SysV SSE+INTEGER)\n");
    { double r = manual_h(hs, hn);
      CHECK(dbits(got_h_s.x) == dbits(hs.x), "s.x bit-exakt (*(RCX)->XMM0)");
      CHECK(got_h_s.y == hs.y, "s.y bit-exakt (8(RCX)->RDI)");
      CHECK(got_h_n == hn, "n bit-exakt (RDX->RSI)");
      CHECK(dbits(r) == dbits(exp_h), "return bit-exakt"); }

    printf("[8] Manuell float-aware: stk (Shadow entfernen, Stack nachbauen)\n");
    { double r = manual_stk_fixed(sa, sb, sc, sd, se, sf, sg, sh);
      CHECK(got_stk_a == sa && got_stk_b == sb && got_stk_c == sc && got_stk_d == sd,
            "a..d bit-exakt");
      CHECK(got_stk_e == se && got_stk_f == sf && got_stk_g == sg, "e..g bit-exakt");
      CHECK(dbits(got_stk_h) == dbits(sh), "h bit-exakt (Stack->XMM0)");
      CHECK(dbits(r) == dbits(exp_stk), "return bit-exakt"); }

    // harte asserts wie gefordert (zusätzlich zum PASS/FAIL-Protokoll)
    assert(fails == 0);
    if (fails == 0) { printf("PASS: alle float/double/struct-by-value Thunks bit-exakt\n"); return 0; }
    printf("FAIL: %d Checks fehlgeschlagen\n", fails);
    return 1;
}
