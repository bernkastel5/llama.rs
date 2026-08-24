// Prefill: N tokens through the same weight matrix.
// A: N separate matvecs (repo). B: one pass over weights, N activations (GEMM).
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define QK_K 256
#define BLK 144
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static inline float f16f(const unsigned char*p){return _cvtsh_ss(p[0]|(p[1]<<8));}
static inline void scale_min(int j,const unsigned char*q,unsigned char*d,unsigned char*m){
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;}else{*d=(q[j+4]&0x0F)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);}}
static inline float hsum(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);
    __m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);return _mm_cvtss_f32(s);}
typedef struct { float d; signed char qs[QK_K]; short bsums[QK_K/16]; } blk_q8K;
__attribute__((target("avx2,fma")))
static void quantize_q8K(const float*x,blk_q8K*y,int nb){
    for(int b=0;b<nb;b++){const float*xb=x+b*QK_K;float amax=0;
        for(int i=0;i<QK_K;i++){float a=fabsf(xb[i]);if(a>amax)amax=a;}
        float d=amax/127.f,id=d?1.f/d:0.f;y[b].d=d;
        for(int i=0;i<QK_K;i++){int v=lrintf(xb[i]*id);y[b].qs[i]=v>127?127:(v<-128?-128:v);}
        for(int i=0;i<QK_K/16;i++){int s=0;for(int j=0;j<16;j++)s+=y[b].qs[i*16+j];y[b].bsums[i]=s;}}}
__attribute__((target("avx2,fma")))
static void dot_q4k_q8k_multi(const unsigned char*row,const blk_q8K*Y,int nb,int N,int stride,float*out){
    const __m256i m4=_mm256_set1_epi8(0xF);
    for(int t=0;t<N;t++)out[t]=0.f;
    for(int b=0;b<nb;b++){
        const unsigned char*blk=row+b*BLK;
        float dw=f16f(blk),dmw=f16f(blk+2);
        unsigned char S[8],M[8];
        for(int g=0;g<8;g++)scale_min(g,blk+4,&S[g],&M[g]);
        const unsigned char*q4=blk+16;
        __m256i W[8];
        for(int g=0;g<4;g++){
            __m256i qb=_mm256_loadu_si256((const __m256i*)(q4+g*32));
            W[2*g]=_mm256_and_si256(qb,m4);
            W[2*g+1]=_mm256_and_si256(_mm256_srli_epi16(qb,4),m4);
        }
        for(int t=0;t<N;t++){
            const blk_q8K*y=Y+(size_t)t*stride+b;
            __m256i sumi=_mm256_setzero_si256();
            int accm=0;
            for(int g=0;g<8;g++){
                __m256i a=_mm256_loadu_si256((const __m256i*)(y->qs+g*32));
                __m256i p=_mm256_maddubs_epi16(W[g],a);
                sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(p,_mm256_set1_epi16((short)S[g])));
                accm+=(int)M[g]*((int)y->bsums[2*g]+(int)y->bsums[2*g+1]);
            }
            __m256 f=_mm256_mul_ps(_mm256_set1_ps(dw*y->d),_mm256_cvtepi32_ps(sumi));
            out[t]+=hsum(f)-dmw*y->d*(float)accm;
        }
    }
}
int main(int argc,char**argv){
    int rows=argc>1?atoi(argv[1]):4864,cols=argc>2?atoi(argv[2]):1024,N=argc>3?atoi(argv[3]):16,iters=argc>4?atoi(argv[4]):5;
    int nb=cols/QK_K; size_t rowb=(size_t)nb*BLK;
    unsigned char*W=aligned_alloc(64,rowb*rows);
    for(size_t i=0;i<rowb*rows;i++)W[i]=rand()&0xff;
    for(int r=0;r<rows;r++)for(int b=0;b<nb;b++){unsigned char*bl=W+(size_t)r*rowb+b*BLK;
        unsigned short d=_cvtss_sh(0.005f,0),dm=_cvtss_sh(0.002f,0);bl[0]=d;bl[1]=d>>8;bl[2]=dm;bl[3]=dm>>8;}
    float*X=aligned_alloc(64,sizeof(float)*(size_t)N*cols);
    for(size_t i=0;i<(size_t)N*cols;i++)X[i]=sinf(i*0.001f);
    blk_q8K*Y=aligned_alloc(64,sizeof(blk_q8K)*(size_t)N*nb);
    float*out=aligned_alloc(64,sizeof(float)*(size_t)rows*N);
    float*tmp=aligned_alloc(64,sizeof(float)*N);
    for(int t=0;t<N;t++)quantize_q8K(X+(size_t)t*cols,Y+(size_t)t*nb,nb);

    // A: N independent matvecs
    double t0=now();
    for(int it=0;it<iters;it++)
      for(int t=0;t<N;t++)
        for(int r=0;r<rows;r++){ dot_q4k_q8k_multi(W+(size_t)r*rowb,Y+(size_t)t*nb,nb,1,nb,tmp); out[r]=tmp[0]; }
    double tA=now()-t0;
    // B: one weight pass, N tokens
    t0=now();
    for(int it=0;it<iters;it++)
      for(int r=0;r<rows;r++) dot_q4k_q8k_multi(W+(size_t)r*rowb,Y,nb,N,nb,out+(size_t)r*N);
    double tB=now()-t0;
    printf("prefill %d tok through %dx%d Q4_K\n",N,rows,cols);
    printf("  A: N x matvec  : %8.3f ms  (%6.3f ms/tok)\n",tA*1000/iters,tA*1000/iters/N);
    printf("  B: batched GEMM: %8.3f ms  (%6.3f ms/tok)  speedup %.2fx\n",tB*1000/iters,tB*1000/iters/N,tA/tB);
    return 0;
}
