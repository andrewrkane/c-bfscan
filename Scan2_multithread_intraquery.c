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

struct arg_struct {
    int topic;
    int startidx;
    int endidx;
    int base;
    heap* h;
};

#define BASESCORE(T) score+=topicsfreq[n][T-2]*( log(1 + tf[base]/(MU * (cf[topics[n][T]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i] + MU)) ); hasScore++;
#define SCORE(T) { BASESCORE(T); continue; }

extern void init_tf(char * data_path);
int num_docs;
int total_terms;
int num_topics;
int search(struct arg_struct *arg) {
  int n = arg->topic;
  int start = arg->startidx;
  int end = arg->endidx;
  heap* h = arg->h;
  heap_create(h,0,NULL);

  int i=0;
  int base = arg->base;
  float score=0;
  int hasScore=0;
  int t;
  float* min_key;
  int* min_val;
  float min_score=0;

  int low=start, high=end;
  if (tweetids[high-1] > topics_time[n]) { high--;
    for (;;) { int p=(low+high)/2; if (p==high) break; if (tweetids[p] > topics_time[n]) high=p; else low=p+1; }
  }

  if ( topics[n][1] == 1 ) {
    for (i=start; likely(i<high); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
      }

      if (unlikely(hasScore)) {
        if (score > min_score) {
          if ( min_score == 0 ) {
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 2 ) {
    for (i=start; likely(i<high); i++) {
      for (int base_end = base+doclengths_ordered[i]; likely(base<base_end); base++) {
        if (unlikely(collection_tf[base] == topics[n][2])) SCORE(2);
        if (unlikely(collection_tf[base] == topics[n][3])) SCORE(3);
      }

      if (unlikely(hasScore)) {
        if (score > min_score) {
          if ( min_score == 0 ) {
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }

  } else if ( topics[n][1] == 3 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 4 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 5 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 6 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 7 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 8 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 9 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else if ( topics[n][1] == 10 ) {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  } else {
    for (i=start; likely(i<high); i++) {
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
            heap_insert(h, scorez, docid);
            int size = heap_size(h);
            if (size>=TOP_K) {
              heap_min(h, (void**)&min_key, (void**)&min_val);
              min_score=*min_key;
            }
          } else {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);

            heap_min(h, (void**)&min_key, (void**)&min_val);
            min_score=*min_key;
          }
        }
        score = 0; hasScore = 0;
      }
    }
  }
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
    
    gettimeofday(&begin, NULL);
    int n;
    for (n=0; n<num_topics; n++) {
      heap h_array[nthreads];
      memset(h_array,0,sizeof(h_array));
      struct threadpool *pool;
      pool = threadpool_init(nthreads);
      int i = 0;
      for (i=0; i<nthreads; i++) {
        struct arg_struct *args = malloc(sizeof *args);
        args->topic = n;
        args->startidx = i*(int)(ceil((double)num_docs / nthreads));
        if ((i+1)*(int)(ceil((double)num_docs / nthreads)) > num_docs) {
          args->endidx = num_docs;
        } else {
          args->endidx = (i+1)*(int)(ceil((double)num_docs / nthreads));
        }
        args->base = termindexes[nthreads-1][i];
        heap h;
        h_array[i] = h;
        args->h = &h_array[i];
        threadpool_add_task(pool,search,args,0);
      }
      threadpool_free(pool,1);

      heap h_merge;
      heap_create(&h_merge,0,NULL);
      float* min_key_merge;
      int* min_val_merge;
      for (i=0; i<nthreads; i++) {
        float* min_key;
        int* min_val;
        while(heap_delmin(&h_array[i], (void**)&min_key, (void**)&min_val)) {
          int size = heap_size(&h_merge);
          if ( size < TOP_K ) {
            heap_insert(&h_merge, min_key, min_val);
          } else {
            heap_min(&h_merge, (void**)&min_key_merge, (void**)&min_val_merge);
            if (*min_key_merge < *min_key) {
              heap_delmin(&h_merge, (void**)&min_key_merge, (void**)&min_val_merge);
              heap_insert(&h_merge, min_key, min_val);
            }
          }
        }
        heap_destroy(&h_array[i]);
      }

      int rank = TOP_K;
      while (heap_delmin(&h_merge, (void**)&min_key_merge, (void**)&min_val_merge)) {
        printf("MB%02d Q0 %ld %d %f Scan2_multithread_intraquery\n", (n+1), tweetids[*min_val_merge], rank, *min_key_merge);
        rank--;
      }
      heap_destroy(&h_merge);
    }
    
    gettimeofday(&end, NULL);
    time_spent = (double)((end.tv_sec * 1000000 + end.tv_usec) - (begin.tv_sec * 1000000 + begin.tv_usec));
    total = total + time_spent / 1000.0;
  }
  printf("Total time = %f ms\n", total/N);
  printf("Time per query = %f ms\n", (total/N)/num_topics);
}
