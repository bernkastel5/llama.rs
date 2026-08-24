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
// AVX2 maddubs baseline
__attribute__((target("avx2,fma")))
static float dot_avx2(const unsigned char*row,const blk_q8K*y,int nb){
    const __m256i m4=_mm256_set1_epi8(0xF); __m256 acc=_mm256_setzero_ps(); float summs=0;
    for(int b=0;b<nb;b++){const unsigned char*blk=row+b*BLK;
        float d=f16f(blk)*y[b].d,dmin=-f16f(blk+2)*y[b].d;
        unsigned char S[8],M[8];for(int g=0;g<8;g++)scale_min(g,blk+4,&S[g],&M[g]);
        const unsigned char*q4=blk+16;__m256i sumi=_mm256_setzero_si256();int accm=0;
        for(int g=0;g<4;g++){
            __m256i qb=_mm256_loadu_si256((const __m256i*)(q4+g*32));
            __m256i wl=_mm256_and_si256(qb,m4),wh=_mm256_and_si256(_mm256_srli_epi16(qb,4),m4);
            __m256i a0=_mm256_loadu_si256((const __m256i*)(y[b].qs+(2*g)*32));
            __m256i a1=_mm256_loadu_si256((const __m256i*)(y[b].qs+(2*g+1)*32));
            sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(_mm256_maddubs_epi16(wl,a0),_mm256_set1_epi16(S[2*g])));
            sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(_mm256_maddubs_epi16(wh,a1),_mm256_set1_epi16(S[2*g+1])));
            accm+=(int)M[2*g]*(y[b].bsums[4*g]+y[b].bsums[4*g+1])+(int)M[2*g+1]*(y[b].bsums[4*g+2]+y[b].bsums[4*g+3]);
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(sumi),acc); summs+=dmin*(float)accm;}
    return hsum(acc)+summs;}
// AVX-512 + VNNI
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
static float dot_avx512(const unsigned char*row,const blk_q8K*y,int nb){
    const __m512i m4=_mm512_set1_epi8(0xF); __m512 acc=_mm512_setzero_ps(); float summs=0;
    for(int b=0;b<nb;b++){const unsigned char*blk=row+b*BLK;
        float d=f16f(blk)*y[b].d,dmin=-f16f(blk+2)*y[b].d;
        unsigned char S[8],M[8];for(int g=0;g<8;g++)scale_min(g,blk+4,&S[g],&M[g]);
        const unsigned char*q4=blk+16;__m512i sumi=_mm512_setzero_si512();int accm=0;
        for(int g=0;g<2;g++){ // 64 bytes -> 128 weights = groups 4g..4g+3
            __m512i qb=_mm512_loadu_si512((const void*)(q4+g*64));
            __m512i wl=_mm512_and_si512(qb,m4);                          // groups 4g, 4g+2 (32B halves)
            __m512i wh=_mm512_and_si512(_mm512_srli_epi16(qb,4),m4);     // groups 4g+1, 4g+3
            __m512i a_l=_mm512_inserti64x4(_mm512_castsi256_si512(
                          _mm256_loadu_si256((const __m256i*)(y[b].qs+(4*g+0)*32))),
                          _mm256_loadu_si256((const __m256i*)(y[b].qs+(4*g+2)*32)),1);
            __m512i a_h=_mm512_inserti64x4(_mm512_castsi256_si512(
                          _mm256_loadu_si256((const __m256i*)(y[b].qs+(4*g+1)*32))),
                          _mm256_loadu_si256((const __m256i*)(y[b].qs+(4*g+3)*32)),1);
            // scale per 32-byte lane: build vector of scales
            __m512i sl=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(S[4*g+0])),_mm256_set1_epi16(S[4*g+2]),1);
            __m512i sh=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(S[4*g+1])),_mm256_set1_epi16(S[4*g+3]),1);
            sumi=_mm512_add_epi32(sumi,_mm512_madd_epi16(_mm512_maddubs_epi16(wl,a_l),sl));
            sumi=_mm512_add_epi32(sumi,_mm512_madd_epi16(_mm512_maddubs_epi16(wh,a_h),sh));
            for(int k=0;k<4;k++){int gg=4*g+k;accm+=(int)M[gg]*(y[b].bsums[2*gg]+y[b].bsums[2*gg+1]);}
        }
        acc=_mm512_fmadd_ps(_mm512_set1_ps(d),_mm512_cvtepi32_ps(sumi),acc); summs+=dmin*(float)accm;}
    return _mm512_reduce_add_ps(acc)+summs;}
int main(int c,char**v){int rows=c>1?atoi(v[1]):4864,cols=c>2?atoi(v[2]):1024,iters=c>3?atoi(v[3]):20;
  int nb=cols/QK_K;size_t rowb=(size_t)nb*BLK;
  unsigned char*W=aligned_alloc(64,rowb*rows);for(size_t i=0;i<rowb*rows;i++)W[i]=rand()&0xff;
  for(int r=0;r<rows;r++)for(int b=0;b<nb;b++){unsigned char*bl=W+(size_t)r*rowb+b*BLK;
     unsigned short d=_cvtss_sh(0.005f,0),dm=_cvtss_sh(0.002f,0);bl[0]=d;bl[1]=d>>8;bl[2]=dm;bl[3]=dm>>8;}
  float*x=aligned_alloc(64,sizeof(float)*cols);for(int i=0;i<cols;i++)x[i]=sinf(i*0.01f);
  blk_q8K*y=aligned_alloc(64,sizeof(blk_q8K)*nb);quantize_q8K(x,y,nb);
  float*o=aligned_alloc(64,sizeof(float)*rows);
  double t=now();for(int it=0;it<iters;it++)for(int r=0;r<rows;r++)o[r]=dot_avx2(W+(size_t)r*rowb,y,nb);double ta=now()-t;float c1=o[0];
  t=now();for(int it=0;it<iters;it++)for(int r=0;r<rows;r++)o[r]=dot_avx512(W+(size_t)r*rowb,y,nb);double tb=now()-t;float c2=o[0];
  printf("%dx%d Q4_K  avx2=%.3f ms  avx512vnni=%.3f ms  speedup %.2fx  (chk %g vs %g)\n",
     rows,cols,ta*1000/iters,tb*1000/iters,ta/tb,c1,c2);
  return 0;}
