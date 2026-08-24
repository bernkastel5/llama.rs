#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static pthread_barrier_t bar; static int NT, ITERS; static volatile long sink;
static void* w(void*a){ for(int i=0;i<ITERS;i++){ pthread_barrier_wait(&bar); sink+=i; pthread_barrier_wait(&bar);} return 0;}
int main(int c,char**v){ NT=c>1?atoi(v[1]):2; ITERS=c>2?atoi(v[2]):200000;
  pthread_barrier_init(&bar,0,NT); pthread_t t[64];
  double s=now();
  for(int i=1;i<NT;i++)pthread_create(&t[i],0,w,0);
  w(0);
  for(int i=1;i<NT;i++)pthread_join(t[i],0);
  double e=now()-s;
  printf("%d threads, %d fork-join rounds: %.3f s total -> %.2f us per join-pair\n",NT,ITERS,e,e*1e6/ITERS);
  printf("  => at 168 matvec barriers/token: %.3f ms/token pure sync overhead\n",e*1e3/ITERS*168);
  return 0;}
