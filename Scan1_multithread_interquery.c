#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

#include "include/constants.h"
#include "include/data.c"
#include "include/heap.c"
#include "include/threadpool.c"

#define unlikely(expr) __builtin_expect(!!(expr),0)
#define likely(expr) __builtin_expect(!!(expr),1)

extern void init_tf(char * data_path);
int num_docs;
int total_terms;
int num_topics;
int search(int n) {
  int i=0;
  int base=0;
  float score=0;

  int t;
  heap h;
  heap_create(&h,0,NULL);

  float* min_key;
  int* min_val;
  float min_score=0;

  int start_doc=0, end_doc=num_docs;
  if (tweetids[end_doc-1] > topics_time[n]) { end_doc--;
    for (;;) { int h=(start_doc+end_doc)/2; if (h==end_doc) break; if (tweetids[h] > topics_time[n]) end_doc=h; else start_doc=h+1; }
  }

  for (i=0; likely(i<end_doc); i++) {
    for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
      for (t=0; t<topics[n][1]; t++) {
        if (unlikely(collection_tf[base] == topics[n][t+2])) {
          score+=topicsfreq[n][t]*( log(1 + tf[base]/(MU * (cf[topics[n][t+2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i] + MU)) );
          break;
        }
      }
    }
    if (unlikely(score > 0)) {
      if (score > min_score) {
        if ( min_score == 0 ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = score;
          heap_insert(&h, scorez, docid);
          int size = heap_size(&h);
          if (size>=TOP_K) {
            heap_min(&h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        } else {
          heap_delmin(&h, (void**)&min_key, (void**)&min_val);
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = score;
          heap_insert(&h, scorez, docid);

          heap_min(&h, (void**)&min_key, (void**)&min_val);
          min_score=*min_key;
        }
      }
      score = 0;
    }
  }

  int rank = TOP_K;
  while (heap_delmin(&h, (void**)&min_key, (void**)&min_val)) {
    printf("MB%02d Q0 %ld %d %f Scan1_multithread_interquery\n", (n+1), tweetids[*min_val], rank, *min_key);
    rank--;
  }

  heap_destroy(&h);
  return 0;
}

int main(int argc, const char* argv[]) {
  if (argc <= 2) {
    printf("PLEASE ENTER DATA PATH AND THREAD NUMBER!\n");
    return 0;
  }
  int nthreads=atoi(argv[2]);
  printf("Number of threads: %d\n", nthreads);
  init_tf(argv[1]);
  double total = 0;
  int N = 3;
  int count;
  for (count = 1; count <= N; count ++) {
    struct timeval begin, end;
    double time_spent;
    
    struct threadpool *pool;
    pool = threadpool_init(nthreads);
    gettimeofday(&begin, NULL);
    int n;
    for (n=0; n<num_topics; n++) {
      threadpool_add_task(pool,search,(void*)n,0);
    }
    threadpool_free(pool,1);
    
    gettimeofday(&end, NULL);
    time_spent = (double)((end.tv_sec * 1000000 + end.tv_usec) - (begin.tv_sec * 1000000 + begin.tv_usec));
    total = total + time_spent / 1000.0;
  }
  printf("Total time = %f ms\n", total/N);
  printf("Throughput: %f qps\n", num_topics/(total/N) * 1000);
}
