// Plan A B4 — CUDA FusionNet inference (Tensor Cores). Convs = im2col + cuBLAS fp16 GEMM with
// fp32 accumulation (CUBLAS_GEMM_DEFAULT_TENSOR_OP -> Tensor Cores). Verified against the PyTorch
// golden (same forward as the CPU reference). This is the runtime inference the recordCnnRefine
// seam will call (the CUDA backend, ADR 0004); the Vulkan/coopmat port mirrors it.
// Build: nvcc -O3 -arch=sm_120 -o cnn_infer_cu cnn_infer.cu -lcublas
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do{ auto e=(x); if(e){printf("cuda err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)
#define FR(p) cudaFreeAsync(p,0)     // stream-ordered free -> pooled reuse (no per-layer sync malloc)

static float h2f(uint16_t h){ uint32_t s=(h&0x8000u)<<16,e=(h>>10)&0x1F,m=h&0x3FF,f;
    if(e==0){ if(m==0)f=s; else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;f=s|(e<<23)|(m<<13);} }
    else if(e==0x1F)f=s|0x7F800000u|(m<<13); else f=s|((e-15+127)<<23)|(m<<13);
    float o; memcpy(&o,&f,4); return o; }

struct HT { std::vector<int> dims; std::vector<__half> d; int n()const{int p=1;for(int x:dims)p*=x;return p;} };
static uint32_t u32(FILE*fp){uint32_t v;fread(&v,4,1,fp);return v;}
static std::vector<__half> rdh(FILE*fp,int n){std::vector<uint16_t> h(n);fread(h.data(),2,n,fp);
    std::vector<__half> o(n);for(int i=0;i<n;i++)o[i]=__float2half(h2f(h[i]));return o;}
static std::map<std::string,HT> load_nvfgw(const char*p){std::map<std::string,HT> M;FILE*fp=fopen(p,"rb");
    char mg[8];fread(mg,1,8,fp);u32(fp);u32(fp);u32(fp);u32(fp);uint32_t nt=u32(fp);
    for(uint32_t i=0;i<nt;i++){uint32_t nl=u32(fp);std::string nm(nl,0);fread(&nm[0],1,nl,fp);
        uint32_t nd=u32(fp);HT t;for(uint32_t j=0;j<nd;j++)t.dims.push_back(u32(fp));t.d=rdh(fp,t.n());M[nm]=t;}
    fclose(fp);return M;}
static HT load_bin(const char*p){FILE*fp=fopen(p,"rb");uint32_t nd=u32(fp);HT t;
    for(uint32_t j=0;j<nd;j++)t.dims.push_back(u32(fp));t.d=rdh(fp,t.n());fclose(fp);return t;}

// GPU tensor [C,H,W] fp16
struct T{__half*d;int C,H,W;};
static T galloc(int C,int H,int W){__half*p;CK(cudaMallocAsync(&p,sizeof(__half)*C*H*W,0));return{p,C,H,W};}
static __half* gup(const std::vector<__half>&v){__half*p;CK(cudaMalloc(&p,sizeof(__half)*v.size()));
    CK(cudaMemcpy(p,v.data(),sizeof(__half)*v.size(),cudaMemcpyHostToDevice));return p;}

__global__ void k_im2col(const __half*in,__half*col,int Ci,int H,int W,int Ho,int Wo,int st){
    int p=blockIdx.x*blockDim.x+threadIdx.x; int N=Ho*Wo; if(p>=N)return; int oy=p/Wo,ox=p%Wo;
    for(int ci=0;ci<Ci;ci++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
        int iy=oy*st-1+ky,ix=ox*st-1+kx; __half v=(iy>=0&&iy<H&&ix>=0&&ix<W)?in[(ci*H+iy)*W+ix]:__float2half(0.f);
        col[((ci*3+ky)*3+kx)*N+p]=v; } }
__global__ void k_gated(const __half*a,const __half*fb,const __half*g,const __half*gb,__half*o,int C,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=C*N)return; int c=i/N;
    float av=__half2float(a[i])+__half2float(fb[c]),gv=__half2float(g[i])+__half2float(gb[c]);
    o[i]=__float2half((av>0?av:0.1f*av)*(1.f/(1.f+expf(-gv)))); }
__global__ void k_bias(const __half*a,const __half*b,__half*o,int C,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=C*N)return; o[i]=__float2half(__half2float(a[i])+__half2float(b[i/N])); }
__global__ void k_up(const __half*in,__half*o,int C,int H,int W,int Ho,int Wo){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=C*Ho*Wo)return; int c=idx/(Ho*Wo),r=idx%(Ho*Wo),oy=r/Wo,ox=r%Wo;
    float iy=(oy+0.5f)*H/Ho-0.5f,ix=(ox+0.5f)*W/Wo-0.5f; int y0=floorf(iy),x0=floorf(ix); float fy=iy-y0,fx=ix-x0;
    int y0c=max(0,min(H-1,y0)),y1c=max(0,min(H-1,y0+1)),x0c=max(0,min(W-1,x0)),x1c=max(0,min(W-1,x0+1));
    const __half*pp=in+(size_t)c*H*W;
    float v=__half2float(pp[y0c*W+x0c])*(1-fx)*(1-fy)+__half2float(pp[y0c*W+x1c])*fx*(1-fy)
           +__half2float(pp[y1c*W+x0c])*(1-fx)*fy+__half2float(pp[y1c*W+x1c])*fx*fy;
    o[idx]=__float2half(v); }

static cublasHandle_t CB;
static std::map<std::string,__half*> WT; static std::map<std::string,std::vector<int>> WD;

// out[Cout,N] = W[Cout,K] @ col[K,N]  via col-major cuBLAS (Tensor Cores, fp32 accum)
static void gemm(const __half*col,const __half*w,__half*out,int Cout,int K,int N){
    float al=1.f,be=0.f;
    CK((cudaError_t)cublasGemmEx(CB,CUBLAS_OP_N,CUBLAS_OP_N,N,Cout,K,&al,
        col,CUDA_R_16F,N, w,CUDA_R_16F,K, &be, out,CUDA_R_16F,N,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }

static T conv_raw(T in,const std::string&wn,int st){
    int Cout=WD[wn][0],K=in.C*9,Ho=(in.H+2-3)/st+1,Wo=(in.W+2-3)/st+1,N=Ho*Wo;
    __half*col; CK(cudaMallocAsync(&col,sizeof(__half)*K*N,0));
    k_im2col<<<(N+255)/256,256>>>(in.d,col,in.C,in.H,in.W,Ho,Wo,st);
    T o=galloc(Cout,Ho,Wo); gemm(col,WT[wn],o.d,Cout,K,N); FR(col); return o; }
static T gated(T in,const std::string&pfx,int st){
    T a=conv_raw(in,pfx+".feat.weight",st), g=conv_raw(in,pfx+".gate.weight",st);
    int C=a.C,N=a.H*a.W; T o=galloc(C,a.H,a.W);
    k_gated<<<(C*N+255)/256,256>>>(a.d,WT[pfx+".feat.bias"],g.d,WT[pfx+".gate.bias"],o.d,C,N);
    FR(a.d);FR(g.d); return o; }
static T seq2(T in,const std::string&e,int st){ T a=gated(in,e+".0",st); T b=gated(a,e+".1",1); FR(a.d); return b; }
static T up(T in,int Ho,int Wo){ T o=galloc(in.C,Ho,Wo); k_up<<<(in.C*Ho*Wo+255)/256,256>>>(in.d,o.d,in.C,in.H,in.W,Ho,Wo); return o; }
static T cat(T a,T b){ int N=a.H*a.W; T o=galloc(a.C+b.C,a.H,a.W);
    cudaMemcpy(o.d,a.d,sizeof(__half)*a.C*N,cudaMemcpyDeviceToDevice);
    cudaMemcpy(o.d+a.C*N,b.d,sizeof(__half)*b.C*N,cudaMemcpyDeviceToDevice); return o; }

int main(int argc,char**argv){
    const char*wp=argc>1?argv[1]:"gold.nvfgw",*xp=argc>2?argv[2]:"gold_x.bin",*yp=argc>3?argv[3]:"gold_y.bin";
    cublasCreate(&CB); cublasSetMathMode(CB,CUBLAS_TENSOR_OP_MATH);
    cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool,0);
    uint64_t thr=~0ull; cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&thr);  // cache freed blocks
    auto M=load_nvfgw(wp); for(auto&kv:M){WT[kv.first]=gup(kv.second.d);WD[kv.first]=kv.second.dims;}
    HT hx=load_bin(xp); T x=galloc(hx.dims[1],hx.dims[2],hx.dims[3]);
    CK(cudaMemcpy(x.d,hx.d.data(),sizeof(__half)*hx.n(),cudaMemcpyHostToDevice));
    auto fwd=[&](){ T s1=seq2(x,"fusion.e1",1),s2=seq2(s1,"fusion.e2",2),s3=seq2(s2,"fusion.e3",2),s4=seq2(s3,"fusion.e4",2);
        T u3=up(s4,s3.H,s3.W),c3=cat(u3,s3),d3=seq2(c3,"fusion.d3",1);
        T u2=up(d3,s2.H,s2.W),c2=cat(u2,s2),d2=seq2(c2,"fusion.d2",1);
        T u1=up(d2,s1.H,s1.W),c1=cat(u1,s1),d1=seq2(c1,"fusion.d1",1);
        T raw=conv_raw(d1,"fusion.out.weight",1); T o=galloc(3,raw.H,raw.W);
        k_bias<<<(3*raw.H*raw.W+255)/256,256>>>(raw.d,WT["fusion.out.bias"],o.d,3,raw.H*raw.W);
        FR(s1.d);FR(s2.d);FR(s3.d);FR(s4.d);FR(u3.d);FR(c3.d);FR(d3.d);
        FR(u2.d);FR(c2.d);FR(d2.d);FR(u1.d);FR(c1.d);FR(d1.d);FR(raw.d); return o; };
    T out=fwd(); CK(cudaDeviceSynchronize());
    std::vector<__half> ho(out.C*out.H*out.W); cudaMemcpy(ho.data(),out.d,sizeof(__half)*ho.size(),cudaMemcpyDeviceToHost);
    HT y=load_bin(yp); double mx=0,sm=0; for(size_t i=0;i<ho.size();i++){double e=fabs(__half2float(ho[i])-__half2float(y.d[i]));mx=e>mx?e:mx;sm+=e;}
    // timing
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b); for(int i=0;i<5;i++){T t=fwd();FR(t.d);}
    cudaEventRecord(a); for(int i=0;i<50;i++){T t=fwd();FR(t.d);} cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    printf("CUDA FusionNet (Tensor Cores) out%dx%dx%d vs golden: max_abs_err=%.4g mean=%.4g | %.3f ms/forward\n",
           out.C,out.H,out.W,mx,sm/ho.size(),ms/50);
    printf("%s\n", mx<3e-2?"B4 CUDA OK (matches PyTorch, Tensor-Core convs)":"MISMATCH");
    return mx<3e-2?0:1;
}
