#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static char*buf; static size_t SZ; static int NT; static double res[64];
static void*rd(void*a){long id=(long)a; size_t per=SZ/NT; char*p=buf+id*per;
  __m256i acc=_mm256_setzero_si256();
  for(size_t i=0;i+32<=per;i+=64) acc=_mm256_add_epi32(acc,_mm256_stream_load_si256((__m256i*)(p+i)));
  res[id]=_mm256_extract_epi32(acc,0); return 0;}
int main(int c,char**v){SZ=(size_t)(c>1?atoi(v[1]):512)*1024*1024;NT=c>2?atoi(v[2]):1;
  buf=aligned_alloc(64,SZ); memset(buf,1,SZ);
  pthread_t t[64]; double s=now();
  for(long i=1;i<NT;i++)pthread_create(&t[i],0,rd,(void*)i); rd(0);
  for(int i=1;i<NT;i++)pthread_join(t[i],0);
  double e=now()-s;
  printf("%d threads: read %.0f MB in %.3f s -> %.2f GB/s\n",NT,SZ/1e6,e,SZ/e/1e9);
  return 0;}
