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

#define BASESCORE(T) score+=topicsfreq[n][T-2]*( log(1 + tf[base]/(MU * (cf[topics[n][T]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i] + MU)) ); hasScore++;
#define SCORE(T) { BASESCORE(T); continue; }

extern void init_tf(char * data_path);
int num_docs;
int total_terms;
int num_topics;
int search(int n) {
  int i=0;

  int base=0;
  float score=0;
  int hasScore=0;

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

  if ( topics[n][1] == 1 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 2 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 3 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 4 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
        if (unlikely(collection_tf[base] == topics[n][5])) SCORE(5);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 5 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
        if (unlikely(collection_tf[base] == topics[n][5])) SCORE(5);
        if (unlikely(collection_tf[base] == topics[n][6])) SCORE(6);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 6 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
        if (unlikely(collection_tf[base] == topics[n][5])) SCORE(5);
        if (unlikely(collection_tf[base] == topics[n][6])) SCORE(6);
        if (unlikely(collection_tf[base] == topics[n][7])) SCORE(7);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 7 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
        if (unlikely(collection_tf[base] == topics[n][5])) SCORE(5);
        if (unlikely(collection_tf[base] == topics[n][6])) SCORE(6);
        if (unlikely(collection_tf[base] == topics[n][7])) SCORE(7);
        if (unlikely(collection_tf[base] == topics[n][8])) SCORE(8);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 8 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
        if (unlikely(collection_tf[base] == topics[n][5])) SCORE(5);
        if (unlikely(collection_tf[base] == topics[n][6])) SCORE(6);
        if (unlikely(collection_tf[base] == topics[n][7])) SCORE(7);
        if (unlikely(collection_tf[base] == topics[n][8])) SCORE(8);
        if (unlikely(collection_tf[base] == topics[n][9])) SCORE(9);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 9 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
        if (unlikely(collection_tf[base] == topics[n][5])) SCORE(5);
        if (unlikely(collection_tf[base] == topics[n][6])) SCORE(6);
        if (unlikely(collection_tf[base] == topics[n][7])) SCORE(7);
        if (unlikely(collection_tf[base] == topics[n][8])) SCORE(8);
        if (unlikely(collection_tf[base] == topics[n][9])) SCORE(9);
        if (unlikely(collection_tf[base] == topics[n][10])) SCORE(10);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 10 ) {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
        if (unlikely(collection_tf[base] == topics[n][4])) SCORE(4);
        if (unlikely(collection_tf[base] == topics[n][5])) SCORE(5);
        if (unlikely(collection_tf[base] == topics[n][6])) SCORE(6);
        if (unlikely(collection_tf[base] == topics[n][7])) SCORE(7);
        if (unlikely(collection_tf[base] == topics[n][8])) SCORE(8);
        if (unlikely(collection_tf[base] == topics[n][9])) SCORE(9);
        if (unlikely(collection_tf[base] == topics[n][10])) SCORE(10);
        if (unlikely(collection_tf[base] == topics[n][11])) SCORE(11);
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  } else {
    for (i=0; likely(i<end_doc); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        for (t=2; t<2+topics[n][1]; t++) {
          if (unlikely(collection_tf[base] == topics[n][t])) { BASESCORE(t); break; }
        }
      }

      if (unlikely(hasScore)) {
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
        score = 0; hasScore = 0;
      }
    }
  }

  int rank = TOP_K;
  while (heap_delmin(&h, (void**)&min_key, (void**)&min_val)) {
    printf("MB%02d Q0 %ld %d %f Scan2_multithread_interquery\n", (n+1), tweetids[*min_val], rank, *min_key);
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
