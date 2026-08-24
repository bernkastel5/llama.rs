// Microbenchmark: current llama.rs AVX2 Q4_K kernel style (f32 FMA, per-8 widening, hsum per group)
// vs llama.cpp style (activations quantized to int8, integer maddubs dot).
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK_K 256
#define BLK 144   // Q4_K block bytes

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static inline float f16f(const unsigned char*p){
    unsigned short b = p[0] | (p[1]<<8);
    return _cvtsh_ss(b);
}
static inline void scale_min(int j,const unsigned char*q,unsigned char*d,unsigned char*m){
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;}
    else{*d=(q[j+4]&0x0F)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}
static inline float hsum(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    __m128 s=_mm_add_ps(lo,hi); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}
static inline __m256i b8_i32(const unsigned char*p){
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p));
}

// ---- A: current repo kernel (f32) ----
__attribute__((target("avx2,fma")))
static float dot_q4k_f32(const unsigned char*row,const float*x,int nb){
    float sum=0.f;
    for(int b=0;b<nb;b++){
        const unsigned char*blk=row+b*BLK;
        float d=f16f(blk), dmin=f16f(blk+2);
        const unsigned char*sc=blk+4; const unsigned char*qs=blk+16;
        const float*xb=x+b*QK_K;
        for(int g=0;g<8;g++){
            unsigned char s,m; scale_min(g,sc,&s,&m);
            __m256 qdot=_mm256_setzero_ps(), xsum=_mm256_setzero_ps();
            for(int c=0;c<4;c++){
                __m256i packed=b8_i32(qs+(g/2)*32+c*8);
                __m256i codes=(g%2==0)?_mm256_and_si256(packed,_mm256_set1_epi32(15))
                                      :_mm256_srli_epi32(packed,4);
                __m256 v=_mm256_loadu_ps(xb+g*32+c*8);
                qdot=_mm256_fmadd_ps(_mm256_cvtepi32_ps(codes),v,qdot);
                xsum=_mm256_add_ps(xsum,v);
            }
            sum += d*(float)s*hsum(qdot) - dmin*(float)m*hsum(xsum);
        }
    }
    return sum;
}

// ---- B: llama.cpp style q4_K x q8_K ----
typedef struct { float d; signed char qs[QK_K]; short bsums[QK_K/16]; } blk_q8K;

__attribute__((target("avx2,fma")))
static void quantize_q8K(const float*x,blk_q8K*y,int nb){
    for(int b=0;b<nb;b++){
        const float*xb=x+b*QK_K; float amax=0.f;
        for(int i=0;i<QK_K;i++){float a=fabsf(xb[i]); if(a>amax)amax=a;}
        float d = amax/127.f, id = d? 1.f/d : 0.f;
        y[b].d=d;
        for(int i=0;i<QK_K;i++){ int v=(int)lrintf(xb[i]*id); if(v>127)v=127; if(v<-128)v=-128; y[b].qs[i]=(signed char)v; }
        for(int i=0;i<QK_K/16;i++){int s=0;for(int j=0;j<16;j++)s+=y[b].qs[i*16+j]; y[b].bsums[i]=(short)s;}
    }
}

__attribute__((target("avx2,fma")))
static float dot_q4k_q8k(const unsigned char*row,const blk_q8K*y,int nb){
    const __m256i m4=_mm256_set1_epi8(0xF);
    const __m128i m32=_mm_set1_epi8(32);
    __m256 acc=_mm256_setzero_ps();
    float summs=0.f;
    for(int b=0;b<nb;b++){
        const unsigned char*blk=row+b*BLK;
        float d=f16f(blk)*y[b].d, dmin=-f16f(blk+2)*y[b].d;
        const unsigned char*sc=blk+4; const unsigned char*q4=blk+16;
        // unpack 12-byte scales into 8 scales + 8 mins (llama.cpp utils)
        unsigned char utmp[16];
        {
            unsigned int u[4]; memcpy(u,sc,12);
            unsigned int u3=u[2];
            u[2]=((u3>>4)&0x0f0f0f0f)|((u[1]>>6)&0x03030303)<<4; // simplified path
            memcpy(utmp,u,12);
            // reference unpack (scalar, exact):
            for(int g=0;g<8;g++){unsigned char s,m;scale_min(g,sc,&s,&m);utmp[g]=s;utmp[8+g]=m;}
        }
        // mins contribution via bsums
        {
            int acc_m=0;
            for(int g=0;g<8;g++){ acc_m += (int)utmp[8+g]*((int)y[b].bsums[2*g]+(int)y[b].bsums[2*g+1]); }
            summs += dmin*(float)acc_m;
        }
        __m256i sumi=_mm256_setzero_si256();
        const signed char*q8=y[b].qs;
        for(int g=0;g<4;g++){
            // 64 weights from 32 bytes -> low nibbles (group 2g), high nibbles (group 2g+1)
            __m256i qbits=_mm256_loadu_si256((const __m256i*)(q4+g*32));
            __m256i wl=_mm256_and_si256(qbits,m4);
            __m256i wh=_mm256_and_si256(_mm256_srli_epi16(qbits,4),m4);
            __m256i a0=_mm256_loadu_si256((const __m256i*)(q8+(2*g)*32));
            __m256i a1=_mm256_loadu_si256((const __m256i*)(q8+(2*g+1)*32));
            __m256i p0=_mm256_maddubs_epi16(wl,a0);   // u8 * i8 -> i16 pairs
            __m256i p1=_mm256_maddubs_epi16(wh,a1);
            __m256i s0=_mm256_set1_epi16((short)utmp[2*g]);
            __m256i s1=_mm256_set1_epi16((short)utmp[2*g+1]);
            sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(p0,s0));
            sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(p1,s1));
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(sumi),acc);
    }
    (void)m32;
    return hsum(acc)+summs;
}

int main(int argc,char**argv){
    int rows = argc>1?atoi(argv[1]):4864;
    int cols = argc>2?atoi(argv[2]):896;
    int iters= argc>3?atoi(argv[3]):20;
    int nb = cols/QK_K;
    if(cols%QK_K){printf("cols must be multiple of 256\n");return 1;}
    size_t rowb = (size_t)nb*BLK;
    unsigned char*W=aligned_alloc(64,rowb*rows);
    float*x=aligned_alloc(64,sizeof(float)*cols);
    float*out=aligned_alloc(64,sizeof(float)*rows);
    srand(1);
    for(size_t i=0;i<rowb*(size_t)rows;i++)W[i]=rand()&0xff;
    // make f16 scales sane
    for(int r=0;r<rows;r++)for(int b=0;b<nb;b++){
        unsigned char*blk=W+(size_t)r*rowb+b*BLK;
        unsigned short d=_cvtss_sh(0.005f,0), dm=_cvtss_sh(0.002f,0);
        blk[0]=d&0xff;blk[1]=d>>8;blk[2]=dm&0xff;blk[3]=dm>>8;
    }
    for(int i=0;i<cols;i++)x[i]=sinf(i*0.01f);
    blk_q8K*y=aligned_alloc(64,sizeof(blk_q8K)*nb);

    double t=now(); float chk=0;
    for(int it=0;it<iters;it++){for(int r=0;r<rows;r++)out[r]=dot_q4k_f32(W+(size_t)r*rowb,x,nb); chk+=out[0];}
    double ta=now()-t;

    t=now(); float chk2=0;
    for(int it=0;it<iters;it++){quantize_q8K(x,y,nb); for(int r=0;r<rows;r++)out[r]=dot_q4k_q8k(W+(size_t)r*rowb,y,nb); chk2+=out[0];}
    double tb=now()-t;

    double bytes=(double)rowb*rows*iters;
    printf("matvec %dx%d Q4_K, %d iters\n",rows,cols,iters);
    printf("  A f32-FMA (repo style) : %8.3f ms/matvec  %6.2f GB/s   chk=%g\n",ta*1000/iters,bytes/ta/1e9,chk);
    printf("  B int8 maddubs (l.cpp) : %8.3f ms/matvec  %6.2f GB/s   chk=%g\n",tb*1000/iters,bytes/tb/1e9,chk2);
    printf("  speedup B/A            : %6.2fx\n",ta/tb);
    return 0;
}
