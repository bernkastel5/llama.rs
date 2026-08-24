#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static atomic_int gen=0, done=0; static int NT,ITERS; static volatile long sink;
static void*w(void*a){int myg=0;
  for(int i=0;i<ITERS;i++){ while(atomic_load_explicit(&gen,memory_order_acquire)==myg) __builtin_ia32_pause(); myg++; sink+=i; atomic_fetch_add_explicit(&done,1,memory_order_release);} return 0;}
int main(int c,char**v){NT=c>1?atoi(v[1]):2;ITERS=c>2?atoi(v[2]):200000;pthread_t t[64];
  for(int i=1;i<NT;i++)pthread_create(&t[i],0,w,0);
  double s=now();
  for(int i=0;i<ITERS;i++){atomic_store_explicit(&done,0,memory_order_release);atomic_fetch_add_explicit(&gen,1,memory_order_release);sink+=i;
     while(atomic_load_explicit(&done,memory_order_acquire)<NT-1)__builtin_ia32_pause();}
  double e=now()-s;
  printf("%d threads spin-barrier: %.2f us/round  => 168 rounds/token = %.3f ms/token\n",NT,e*1e6/ITERS,e*1e3/ITERS*168);
  for(int i=1;i<NT;i++)pthread_join(t[i],0);
  return 0;}
