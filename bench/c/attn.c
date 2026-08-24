#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

// A: strictly-ordered scalar reduction (what Rust `.map().sum::<f32>()` compiles to)
__attribute__((target("avx2,fma"),optimize("no-fast-math")))
static float dot_scalar_ordered(const float*a,const float*b,int n){
    volatile float s=0.f; // force strict ordering, no vectorization
    float acc=0.f;
    for(int i=0;i<n;i++) acc+=a[i]*b[i];
    s=acc; return s;
}
__attribute__((target("avx2,fma")))
static float dot_avx(const float*a,const float*b,int n){
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
    int i=0;
    for(;i+16<=n;i+=16){
        a0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),a0);
        a1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),a1);
    }
    __m256 v=_mm256_add_ps(a0,a1);
    __m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);
    __m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);
    float r=_mm_cvtss_f32(s);
    for(;i<n;i++)r+=a[i]*b[i];
    return r;
}
// f16 KV cache variant
__attribute__((target("avx2,fma,f16c")))
static float dot_avx_f16(const unsigned short*a,const float*b,int n){
    __m256 acc=_mm256_setzero_ps();
    int i=0;
    for(;i+8<=n;i+=8){
        __m256 w=_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a+i)));
        acc=_mm256_fmadd_ps(w,_mm256_loadu_ps(b+i),acc);
    }
    __m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1);
    __m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}

int main(int argc,char**argv){
    int ctx=argc>1?atoi(argv[1]):512;
    int heads=14,kvheads=2,hd=64,layers=24,iters=argc>2?atoi(argv[2]):20;
    int kvw=kvheads*hd;
    float*K=aligned_alloc(64,sizeof(float)*(size_t)ctx*kvw);
    unsigned short*K16=aligned_alloc(64,sizeof(unsigned short)*(size_t)ctx*kvw);
    float*q=aligned_alloc(64,sizeof(float)*heads*hd);
    float*sc=aligned_alloc(64,sizeof(float)*ctx);
    for(size_t i=0;i<(size_t)ctx*kvw;i++){K[i]=sinf(i*0.001f);K16[i]=_cvtss_sh(K[i],0);}
    for(int i=0;i<heads*hd;i++)q[i]=cosf(i*0.01f);

    double t,tA,tB,tC; float chk=0;
    t=now();
    for(int it=0;it<iters;it++)for(int l=0;l<layers;l++)for(int h=0;h<heads;h++){
        const float*qh=q+h*hd; int kvh=h/(heads/kvheads);
        for(int p=0;p<ctx;p++) sc[p]=dot_scalar_ordered(qh,K+(size_t)p*kvw+kvh*hd,hd);
        chk+=sc[0];
    }
    tA=now()-t;
    t=now();
    for(int it=0;it<iters;it++)for(int l=0;l<layers;l++)for(int h=0;h<heads;h++){
        const float*qh=q+h*hd; int kvh=h/(heads/kvheads);
        for(int p=0;p<ctx;p++) sc[p]=dot_avx(qh,K+(size_t)p*kvw+kvh*hd,hd);
        chk+=sc[0];
    }
    tB=now()-t;
    t=now();
    for(int it=0;it<iters;it++)for(int l=0;l<layers;l++)for(int h=0;h<heads;h++){
        const float*qh=q+h*hd; int kvh=h/(heads/kvheads);
        for(int p=0;p<ctx;p++) sc[p]=dot_avx_f16(K16+(size_t)p*kvw+kvh*hd,qh,hd);
        chk+=sc[0];
    }
    tC=now()-t;
    printf("QK^T all %d layers, ctx=%d (per token, 1 thread)  chk=%g\n",layers,ctx,chk);
    printf("  scalar ordered (repo) : %8.3f ms/token\n",tA*1000/iters);
    printf("  avx2 f32 KV           : %8.3f ms/token  (%.2fx)\n",tB*1000/iters,tA/tB);
    printf("  avx2 f16 KV           : %8.3f ms/token  (%.2fx)\n",tC*1000/iters,tA/tC);
    return 0;
}
