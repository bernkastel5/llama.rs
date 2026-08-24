// Full Qwen2.5-0.5B decode-step simulation, Q4_K weights.
// A = repo design (f32 unpack kernel, ordered-scalar attention, per-matvec fork/join)
// B = proposed (q8 activations + maddubs, AVX2 attention, persistent spin barrier)
#include <immintrin.h>
#include <pthread.h>
#include <stdatomic.h>
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
static inline float hs8(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);
    __m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);return _mm_cvtss_f32(s);}
typedef struct{float d;signed char qs[QK_K];short bsums[QK_K/16];}blk_q8K;
static inline __m256i b8(const unsigned char*p){return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p));}

__attribute__((target("avx2,fma")))
static float A_dot(const unsigned char*row,const float*x,int nb){
    float sum=0;
    for(int b=0;b<nb;b++){const unsigned char*blk=row+b*BLK;
        float d=f16f(blk),dmin=f16f(blk+2);const unsigned char*qs=blk+16;const float*xb=x+b*QK_K;
        for(int g=0;g<8;g++){unsigned char s,m;scale_min(g,blk+4,&s,&m);
            __m256 qd=_mm256_setzero_ps(),xs=_mm256_setzero_ps();
            for(int c=0;c<4;c++){__m256i p=b8(qs+(g/2)*32+c*8);
                __m256i cd=(g%2==0)?_mm256_and_si256(p,_mm256_set1_epi32(15)):_mm256_srli_epi32(p,4);
                __m256 v=_mm256_loadu_ps(xb+g*32+c*8);
                qd=_mm256_fmadd_ps(_mm256_cvtepi32_ps(cd),v,qd);xs=_mm256_add_ps(xs,v);}
            sum+=d*s*hs8(qd)-dmin*m*hs8(xs);}}
    return sum;}
__attribute__((target("avx2,fma")))
static void B_quant(const float*x,blk_q8K*y,int nb){
    for(int b=0;b<nb;b++){const float*xb=x+b*QK_K;float amax=0;
        for(int i=0;i<QK_K;i++){float a=fabsf(xb[i]);if(a>amax)amax=a;}
        float d=amax/127.f,id=d?1.f/d:0.f;y[b].d=d;
        for(int i=0;i<QK_K;i++){int v=lrintf(xb[i]*id);y[b].qs[i]=v>127?127:(v<-128?-128:v);}
        for(int i=0;i<QK_K/16;i++){int s=0;for(int j=0;j<16;j++)s+=y[b].qs[i*16+j];y[b].bsums[i]=s;}}}
__attribute__((target("avx2,fma")))
static float B_dot(const unsigned char*row,const blk_q8K*y,int nb){
    const __m256i m4=_mm256_set1_epi8(0xF);__m256 acc=_mm256_setzero_ps();float summs=0;
    for(int b=0;b<nb;b++){const unsigned char*blk=row+b*BLK;
        float d=f16f(blk)*y[b].d,dmin=-f16f(blk+2)*y[b].d;
        unsigned char S[8],M[8];for(int g=0;g<8;g++)scale_min(g,blk+4,&S[g],&M[g]);
        const unsigned char*q4=blk+16;__m256i sumi=_mm256_setzero_si256();int accm=0;
        for(int g=0;g<4;g++){__m256i qb=_mm256_loadu_si256((const __m256i*)(q4+g*32));
            __m256i wl=_mm256_and_si256(qb,m4),wh=_mm256_and_si256(_mm256_srli_epi16(qb,4),m4);
            sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(_mm256_maddubs_epi16(wl,
                 _mm256_loadu_si256((const __m256i*)(y[b].qs+(2*g)*32))),_mm256_set1_epi16(S[2*g])));
            sumi=_mm256_add_epi32(sumi,_mm256_madd_epi16(_mm256_maddubs_epi16(wh,
                 _mm256_loadu_si256((const __m256i*)(y[b].qs+(2*g+1)*32))),_mm256_set1_epi16(S[2*g+1])));
            accm+=(int)M[2*g]*(y[b].bsums[4*g]+y[b].bsums[4*g+1])+(int)M[2*g+1]*(y[b].bsums[4*g+2]+y[b].bsums[4*g+3]);}
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(sumi),acc);summs+=dmin*(float)accm;}
    return hs8(acc)+summs;}
__attribute__((target("avx2,fma")))
static float dotf(const float*a,const float*b,int n){__m256 s=_mm256_setzero_ps();int i=0;
    for(;i+8<=n;i+=8)s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hs8(s);for(;i<n;i++)r+=a[i]*b[i];return r;}
static float dots(const float*a,const float*b,int n){float r=0;for(int i=0;i<n;i++)r+=a[i]*b[i];return r;}

// model dims (Qwen2.5-0.5B)
#define HID 896
#define INT_ 4864
#define NH 14
#define NKV 2
#define HD 64
#define NL 24
#define VOC 151936
typedef struct{unsigned char*q,*k,*v,*o,*g,*u,*dn;}Layer;
static Layer L[NL]; static unsigned char*LMH;
static float *hid,*qb_,*kb,*vb,*att,*proj,*gt,*up,*dwn,*logits,*Kc,*Vc,*scr;
static blk_q8K *y_hid,*y_int,*y_att;
static int NT,CTX,MODE; static atomic_int gen,doneflag; static volatile int stop_;

static unsigned char* mkw(int rows,int cols){int nb=cols/QK_K;size_t rb=(size_t)nb*BLK;
    unsigned char*w=aligned_alloc(64,rb*rows);
    for(size_t i=0;i<rb*rows;i++)w[i]=rand()&0xff;
    for(int r=0;r<rows;r++)for(int b=0;b<nb;b++){unsigned char*bl=w+(size_t)r*rb+b*BLK;
        unsigned short d=_cvtss_sh(0.005f,0),dm=_cvtss_sh(0.002f,0);bl[0]=d;bl[1]=d>>8;bl[2]=dm;bl[3]=dm>>8;}
    return w;}
static void mv_range(const unsigned char*W,int cols,const float*x,const blk_q8K*y,float*out,int r0,int r1){
    int nb=cols/QK_K;size_t rb=(size_t)nb*BLK;
    if(MODE==0)for(int r=r0;r<r1;r++)out[r]=A_dot(W+(size_t)r*rb,x,nb);
    else for(int r=r0;r<r1;r++)out[r]=B_dot(W+(size_t)r*rb,y,nb);}
static void split(int rows,int id,int*a,int*b){int per=(rows+NT-1)/NT;*a=id*per;*b=(id+1)*per;if(*b>rows)*b=rows;}

static void worker_body(int id){
    int a,b;
    for(int l=0;l<NL;l++){
        split(HID,id,&a,&b); mv_range(L[l].q,HID,hid,y_hid,qb_,a,b);
        split(NKV*HD,id,&a,&b); mv_range(L[l].k,HID,hid,y_hid,kb,a,b);
        split(NKV*HD,id,&a,&b); mv_range(L[l].v,HID,hid,y_hid,vb,a,b);
        // attention: split heads
        int hper=(NH+NT-1)/NT,h0=id*hper,h1=h0+hper>NH?NH:h0+hper;
        for(int h=h0;h<h1;h++){int kvh=h/(NH/NKV);const float*qh=qb_+h*HD;
            for(int p=0;p<CTX;p++){const float*kp=Kc+((size_t)l*CTX+p)*(NKV*HD)+kvh*HD;
                scr[h*CTX+p]=(MODE==0)?dots(qh,kp,HD):dotf(qh,kp,HD);}
            float*ao=att+h*HD;memset(ao,0,HD*4);
            for(int p=0;p<CTX;p++){const float*vp=Vc+((size_t)l*CTX+p)*(NKV*HD)+kvh*HD;float w=scr[h*CTX+p]*0.001f;
                if(MODE==0){for(int i=0;i<HD;i++)ao[i]+=w*vp[i];}
                else{__m256 wv=_mm256_set1_ps(w);for(int i=0;i<HD;i+=8)
                     _mm256_storeu_ps(ao+i,_mm256_fmadd_ps(wv,_mm256_loadu_ps(vp+i),_mm256_loadu_ps(ao+i)));}}}
        split(HID,id,&a,&b); mv_range(L[l].o,HID,att,y_att,proj,a,b);
        split(INT_,id,&a,&b); mv_range(L[l].g,HID,hid,y_hid,gt,a,b);
        split(INT_,id,&a,&b); mv_range(L[l].u,HID,hid,y_hid,up,a,b);
        split(HID,id,&a,&b); mv_range(L[l].dn,INT_,gt,y_int,dwn,a,b);
    }
    split(VOC,id,&a,&b); mv_range(LMH,HID,hid,y_hid,logits,a,b);
}
static void*wthread(void*p){long id=(long)p;int myg=0;
    for(;;){while(atomic_load_explicit(&gen,memory_order_acquire)==myg){if(stop_)return 0;__builtin_ia32_pause();}
        myg++; worker_body(id); atomic_fetch_add_explicit(&doneflag,1,memory_order_release);} }
int main(int c,char**v){
    NT=c>1?atoi(v[1]):2; CTX=c>2?atoi(v[2]):128; int iters=c>3?atoi(v[3]):3;
    srand(7);
    for(int l=0;l<NL;l++){L[l].q=mkw(HID,HID);L[l].k=mkw(NKV*HD,HID);L[l].v=mkw(NKV*HD,HID);
        L[l].o=mkw(HID,HID);L[l].g=mkw(INT_,HID);L[l].u=mkw(INT_,HID);L[l].dn=mkw(HID,INT_);}
    LMH=mkw(VOC,HID);
    hid=aligned_alloc(64,HID*4);qb_=aligned_alloc(64,HID*4);kb=aligned_alloc(64,NKV*HD*4);
    vb=aligned_alloc(64,NKV*HD*4);att=aligned_alloc(64,HID*4);proj=aligned_alloc(64,HID*4);
    gt=aligned_alloc(64,INT_*4);up=aligned_alloc(64,INT_*4);dwn=aligned_alloc(64,HID*4);
    logits=aligned_alloc(64,(size_t)VOC*4);scr=aligned_alloc(64,(size_t)NH*CTX*4);
    Kc=aligned_alloc(64,(size_t)NL*CTX*NKV*HD*4);Vc=aligned_alloc(64,(size_t)NL*CTX*NKV*HD*4);
    for(int i=0;i<HID;i++)hid[i]=sinf(i*0.01f);for(int i=0;i<INT_;i++)gt[i]=cosf(i*0.01f);
    for(size_t i=0;i<(size_t)NL*CTX*NKV*HD;i++){Kc[i]=sinf(i*0.001f);Vc[i]=cosf(i*0.001f);}
    y_hid=aligned_alloc(64,sizeof(blk_q8K)*(HID/QK_K+1));
    y_int=aligned_alloc(64,sizeof(blk_q8K)*(INT_/QK_K+1));
    y_att=aligned_alloc(64,sizeof(blk_q8K)*(HID/QK_K+1));
    printf("Qwen2.5-0.5B Q4_K decode sim, %d threads, ctx=%d\n",NT,CTX);
    double res[2];
    for(MODE=0;MODE<2;MODE++){
        if(MODE==1){B_quant(hid,y_hid,HID/QK_K);B_quant(gt,y_int,INT_/QK_K);B_quant(att,y_att,HID/QK_K);}
        stop_=0;gen=0;doneflag=0;pthread_t th[64];
        for(long i=1;i<NT;i++)pthread_create(&th[i],0,wthread,(void*)i);
        worker_body(0); // warm
        double t=now();
        for(int it=0;it<iters;it++){
            atomic_store_explicit(&doneflag,0,memory_order_release);
            atomic_fetch_add_explicit(&gen,1,memory_order_release);
            worker_body(0);
            while(atomic_load_explicit(&doneflag,memory_order_acquire)<NT-1)__builtin_ia32_pause();}
        res[MODE]=(now()-t)*1000/iters;
        stop_=1;atomic_fetch_add(&gen,1);
        for(int i=1;i<NT;i++)pthread_join(th[i],0);
        printf("  %s: %8.2f ms/token = %6.2f tok/s\n",MODE?"B proposed":"A current ",res[MODE],1000/res[MODE]);
    }
    printf("  speedup: %.2fx\n",res[0]/res[1]);
    return 0;}
