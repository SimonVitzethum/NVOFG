// Plan A B4 — CPU reference inference of the FusionNet from a .nvfgw file.
// Proves the export format + the inference graph are correct (golden test vs PyTorch) BEFORE the
// GPU (coopmat/CUDA) kernels. The GPU backend optimizes this exact, verified forward.
// Build: c++ -O2 -o cnn_infer cnn_infer.cpp
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

struct Tensor { std::vector<int> dims; std::vector<float> d; int n() const { int p=1; for(int x:dims)p*=x; return p; } };

static float h2f(uint16_t h){
    uint32_t s=(h&0x8000u)<<16, e=(h>>10)&0x1F, m=h&0x3FF, f;
    if(e==0){ if(m==0) f=s; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3FF; f=s|(e<<23)|(m<<13);} }
    else if(e==0x1F) f=s|0x7F800000u|(m<<13);
    else f=s|((e-15+127)<<23)|(m<<13);
    float o; std::memcpy(&o,&f,4); return o;
}
static std::vector<float> rd_fp16(FILE* fp, int n){ std::vector<uint16_t> h(n); fread(h.data(),2,n,fp); std::vector<float> o(n); for(int i=0;i<n;i++)o[i]=h2f(h[i]); return o; }
static uint32_t u32(FILE* fp){ uint32_t v; fread(&v,4,1,fp); return v; }

static std::map<std::string,Tensor> load_nvfgw(const char* p){
    std::map<std::string,Tensor> M; FILE* fp=fopen(p,"rb"); char magic[8]; fread(magic,1,8,fp);
    u32(fp);u32(fp);u32(fp);u32(fp);           // version, base, cin, mode
    uint32_t nt=u32(fp);
    for(uint32_t i=0;i<nt;i++){ uint32_t nl=u32(fp); std::string nm(nl,0); fread(&nm[0],1,nl,fp);
        uint32_t nd=u32(fp); Tensor t; for(uint32_t j=0;j<nd;j++)t.dims.push_back(u32(fp));
        t.d=rd_fp16(fp,t.n()); M[nm]=t; }
    fclose(fp); return M;
}
static Tensor load_bin(const char* p){ FILE* fp=fopen(p,"rb"); uint32_t nd=u32(fp); Tensor t;
    for(uint32_t j=0;j<nd;j++)t.dims.push_back(u32(fp)); t.d=rd_fp16(fp,t.n()); fclose(fp); return t; }

// conv2d cross-correlation, padding 1, kernel 3. in[Ci,H,W] w[Co,Ci,3,3] b[Co] -> [Co,Hout,Wout]
static Tensor conv3(const Tensor& in, const Tensor& w, const Tensor& b, int stride){
    int Ci=in.dims[0],H=in.dims[1],W=in.dims[2], Co=w.dims[0];
    int Ho=(H+2-3)/stride+1, Wo=(W+2-3)/stride+1;
    Tensor o; o.dims={Co,Ho,Wo}; o.d.assign((size_t)Co*Ho*Wo,0.f);
    for(int co=0;co<Co;co++) for(int oy=0;oy<Ho;oy++) for(int ox=0;ox<Wo;ox++){
        float s=b.d[co];
        for(int ci=0;ci<Ci;ci++) for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++){
            int iy=oy*stride-1+ky, ix=ox*stride-1+kx;
            if(iy>=0&&iy<H&&ix>=0&&ix<W)
                s+=in.d[((size_t)ci*H+iy)*W+ix]*w.d[(((size_t)co*Ci+ci)*3+ky)*3+kx];
        }
        o.d[((size_t)co*Ho+oy)*Wo+ox]=s;
    }
    return o;
}
static inline float leaky(float x){ return x>0?x:0.1f*x; }
static inline float sigm(float x){ return 1.f/(1.f+std::exp(-x)); }

// GatedConv: leaky(feat(x)) * sigmoid(gate(x))
static Tensor gated(const Tensor& in, std::map<std::string,Tensor>& M, const std::string& pfx, int stride){
    Tensor a=conv3(in,M[pfx+".feat.weight"],M[pfx+".feat.bias"],stride);
    Tensor g=conv3(in,M[pfx+".gate.weight"],M[pfx+".gate.bias"],stride);
    for(size_t i=0;i<a.d.size();i++) a.d[i]=leaky(a.d[i])*sigm(g.d[i]);
    return a;
}
static Tensor seq2(const Tensor& in, std::map<std::string,Tensor>& M, const std::string& e, int stride){
    return gated(gated(in,M,e+".0",stride),M,e+".1",1);
}
static Tensor upsample(const Tensor& in,int Ho,int Wo){    // bilinear, align_corners=False
    int C=in.dims[0],H=in.dims[1],W=in.dims[2]; Tensor o; o.dims={C,Ho,Wo}; o.d.assign((size_t)C*Ho*Wo,0.f);
    for(int oy=0;oy<Ho;oy++){ float iy=(oy+0.5f)*H/Ho-0.5f; int y0=(int)std::floor(iy); float fy=iy-y0;
        int y0c=y0<0?0:(y0>H-1?H-1:y0), y1c=(y0+1)<0?0:((y0+1)>H-1?H-1:(y0+1));
        for(int ox=0;ox<Wo;ox++){ float ix=(ox+0.5f)*W/Wo-0.5f; int x0=(int)std::floor(ix); float fx=ix-x0;
            int x0c=x0<0?0:(x0>W-1?W-1:x0), x1c=(x0+1)<0?0:((x0+1)>W-1?W-1:(x0+1));
            for(int c=0;c<C;c++){ const float* p=&in.d[(size_t)c*H*W];
                float v=p[y0c*W+x0c]*(1-fx)*(1-fy)+p[y0c*W+x1c]*fx*(1-fy)+p[y1c*W+x0c]*(1-fx)*fy+p[y1c*W+x1c]*fx*fy;
                o.d[((size_t)c*Ho+oy)*Wo+ox]=v; } } }
    return o;
}
static Tensor cat(const Tensor& a,const Tensor& b){ int H=a.dims[1],W=a.dims[2]; Tensor o; o.dims={a.dims[0]+b.dims[0],H,W};
    o.d=a.d; o.d.insert(o.d.end(),b.d.begin(),b.d.end()); return o; }

int main(int argc,char**argv){
    const char* wp=argc>1?argv[1]:"gold.nvfgw"; const char* xp=argc>2?argv[2]:"gold_x.bin"; const char* yp=argc>3?argv[3]:"gold_y.bin";
    auto M=load_nvfgw(wp); Tensor x=load_bin(xp); x.dims={x.dims[1],x.dims[2],x.dims[3]}; // drop batch
    Tensor s1=seq2(x,M,"fusion.e1",1), s2=seq2(s1,M,"fusion.e2",2), s3=seq2(s2,M,"fusion.e3",2), s4=seq2(s3,M,"fusion.e4",2);
    Tensor d3=seq2(cat(upsample(s4,s3.dims[1],s3.dims[2]),s3),M,"fusion.d3",1);
    Tensor d2=seq2(cat(upsample(d3,s2.dims[1],s2.dims[2]),s2),M,"fusion.d2",1);
    Tensor d1=seq2(cat(upsample(d2,s1.dims[1],s1.dims[2]),s1),M,"fusion.d1",1);
    Tensor out=conv3(d1,M["fusion.out.weight"],M["fusion.out.bias"],1);
    Tensor y=load_bin(yp);
    double maxerr=0,sum=0; for(size_t i=0;i<out.d.size();i++){ double e=std::fabs(out.d[i]-y.d[i]); maxerr=e>maxerr?e:maxerr; sum+=e; }
    printf("tensors=%zu  out%dx%dx%d  vs golden  max_abs_err=%.4g  mean=%.4g\n",M.size(),out.dims[0],out.dims[1],out.dims[2],maxerr,sum/out.d.size());
    printf("%s\n", maxerr<2e-2 ? "B4 CPU REFERENCE OK (matches PyTorch within fp16 tolerance)" : "MISMATCH");
    return maxerr<2e-2?0:1;
}
