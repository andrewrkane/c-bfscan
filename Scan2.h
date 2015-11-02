#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "include/constants.h"
#include "include/data.c"
#include "include/heap.c"

#define SCANNAME "Scan2"

#define unlikely(expr) __builtin_expect(!!(expr),0)
#define likely(expr) __builtin_expect(!!(expr),1)

extern void init_tf(char * data_path);
int num_docs;
int total_terms;
int num_topics;

struct arg_struct {
    int topic;
    int startidx;
    int endidx;
    int base;
    heap* h;
    int done;
};

#define BASESCORE(T) score+=topicsfreq[n][T-2]*( log(1 + tf[base]/(MU * (cf[topics[n][T]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i] + MU)) ); hasScore++;
#define SCORE(T) { BASESCORE(T); continue; }

#define PREFETCHC //__builtin_prefetch(&collection_tf[base+1024]);

int scansearch(struct arg_struct *arg) {
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
      PREFETCHC
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
  arg->done=1;
  return 0;
}

