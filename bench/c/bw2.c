#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static char*buf; static size_t SZ; static int NT,REP; static volatile long sink[64];
static void*rd(void*a){long id=(long)a; size_t per=SZ/NT; char*p=buf+id*per; long s=0;
  for(int r=0;r<REP;r++){ __m256i acc=_mm256_setzero_si256();
    for(size_t i=0;i+256<=per;i+=256){
      acc=_mm256_add_epi32(acc,_mm256_load_si256((__m256i*)(p+i)));
      acc=_mm256_add_epi32(acc,_mm256_load_si256((__m256i*)(p+i+64)));
      acc=_mm256_add_epi32(acc,_mm256_load_si256((__m256i*)(p+i+128)));
      acc=_mm256_add_epi32(acc,_mm256_load_si256((__m256i*)(p+i+192)));}
    s+=_mm256_extract_epi32(acc,0);} sink[id]=s; return 0;}
int main(int c,char**v){SZ=(size_t)(c>1?atoi(v[1]):1024)*1024*1024;NT=c>2?atoi(v[2]):1;REP=c>3?atoi(v[3]):3;
  buf=aligned_alloc(64,SZ); memset(buf,1,SZ);
  pthread_t t[64]; double s=now();
  for(long i=1;i<NT;i++)pthread_create(&t[i],0,rd,(void*)i); rd(0);
  for(int i=1;i<NT;i++)pthread_join(t[i],0);
  double e=now()-s;
  // only 1/4 of each 256B stride actually touched? no - we load 4x32B out of 256B => 1/2... compute touched bytes
  double touched=(double)SZ*REP*(128.0/256.0);
  printf("%d thr: %.2f GB/s effective (touched %.0f MB, %.3f s)\n",NT,touched/e/1e9,touched/1e6,e);
  return 0;}
