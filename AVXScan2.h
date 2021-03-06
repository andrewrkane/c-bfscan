#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "immintrin.h"
#include "include/constants.h"
#include "include/data.c"
#include "include/heap.c"

#define SCANNAME "AVXScan2"

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
  
  int i=0, j=0;
  int base = arg->base;
  float score;
  int len1, len2, len3, len4, len5, len6;
  int indexsum;
  __m256i collect_vec, mask;
  __m256 score_vec, t1, t2;
  __m128 t3, t4;
  int pos;
  
  int t;
  int jump = 6;
  
  float* min_key;
  int* min_val;
  float score_array[8];
  float scores[jump];
  
  int low=start, high=end;
  if (tweetids[high-1] > topics_time[n]) { high--;
    for (;;) { int p=(low+high)/2; if (p==high) break; if (tweetids[p] > topics_time[n]) high=p; else low=p+1; }
  }
  
  if ( topics[n][1] == 1 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  } else if ( topics[n][1] == 2 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  } else if ( topics[n][1] == 3 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  } else if ( topics[n][1] == 4 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    __m256i query_vec_4 = _mm256_set1_epi32(topics[n][5]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_4);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  } else if ( topics[n][1] == 5 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    __m256i query_vec_4 = _mm256_set1_epi32(topics[n][5]);
    __m256i query_vec_5 = _mm256_set1_epi32(topics[n][6]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_4);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_5);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  } else if ( topics[n][1] == 6 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    __m256i query_vec_4 = _mm256_set1_epi32(topics[n][5]);
    __m256i query_vec_5 = _mm256_set1_epi32(topics[n][6]);
    __m256i query_vec_6 = _mm256_set1_epi32(topics[n][7]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_4);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_5);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_6);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  } else if ( topics[n][1] == 7 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    __m256i query_vec_4 = _mm256_set1_epi32(topics[n][5]);
    __m256i query_vec_5 = _mm256_set1_epi32(topics[n][6]);
    __m256i query_vec_6 = _mm256_set1_epi32(topics[n][7]);
    __m256i query_vec_7 = _mm256_set1_epi32(topics[n][8]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_4);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_5);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_6);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_7);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  }  else if ( topics[n][1] == 8 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    __m256i query_vec_4 = _mm256_set1_epi32(topics[n][5]);
    __m256i query_vec_5 = _mm256_set1_epi32(topics[n][6]);
    __m256i query_vec_6 = _mm256_set1_epi32(topics[n][7]);
    __m256i query_vec_7 = _mm256_set1_epi32(topics[n][8]);
    __m256i query_vec_8 = _mm256_set1_epi32(topics[n][9]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_4);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_5);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_6);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_7);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_8);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  }  else if ( topics[n][1] == 9 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    __m256i query_vec_4 = _mm256_set1_epi32(topics[n][5]);
    __m256i query_vec_5 = _mm256_set1_epi32(topics[n][6]);
    __m256i query_vec_6 = _mm256_set1_epi32(topics[n][7]);
    __m256i query_vec_7 = _mm256_set1_epi32(topics[n][8]);
    __m256i query_vec_8 = _mm256_set1_epi32(topics[n][9]);
    __m256i query_vec_9 = _mm256_set1_epi32(topics[n][10]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_4);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_5);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_6);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_7);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_8);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_9);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  }  else if ( topics[n][1] == 10 ) {
    __m256i query_vec_1 = _mm256_set1_epi32(topics[n][2]);
    __m256i query_vec_2 = _mm256_set1_epi32(topics[n][3]);
    __m256i query_vec_3 = _mm256_set1_epi32(topics[n][4]);
    __m256i query_vec_4 = _mm256_set1_epi32(topics[n][5]);
    __m256i query_vec_5 = _mm256_set1_epi32(topics[n][6]);
    __m256i query_vec_6 = _mm256_set1_epi32(topics[n][7]);
    __m256i query_vec_7 = _mm256_set1_epi32(topics[n][8]);
    __m256i query_vec_8 = _mm256_set1_epi32(topics[n][9]);
    __m256i query_vec_9 = _mm256_set1_epi32(topics[n][10]);
    __m256i query_vec_10 = _mm256_set1_epi32(topics[n][11]);
    for (i=start; i<end; i+=jump) {
      len1 = doclengths_ordered_padding[i];
      len2 = doclengths_ordered_padding[i+1];
      len3 = doclengths_ordered_padding[i+2];
      len4 = doclengths_ordered_padding[i+3];
      len5 = doclengths_ordered_padding[i+4];
      len6 = doclengths_ordered_padding[i+5];
      indexsum = len1 + len2 + len3 + len4 + len5 + len6;
      memset(scores, 0, sizeof(scores));
      for (j=0; j<indexsum; j+=8) {
        pos = doc_pos[base+j];
        collect_vec = _mm256_loadu_si256(&collection_tf_padding[base+j]);
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_1);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_2);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][3]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_3);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][4]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_4);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][5]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_5);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][6]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_6);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][7]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_7);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][8]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_8);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][9]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_9);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][10]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
        mask = _mm256_cmpeq_epi32(collect_vec, query_vec_10);
        if (unlikely(_mm256_movemask_epi8(mask) != 0)) {
          memset(score_array, 0.0, sizeof(score_array));
          score_array[0] = log(1 + tf_padding[base+j]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[1] = log(1 + tf_padding[base+j+1]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[2] = log(1 + tf_padding[base+j+2]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[3] = log(1 + tf_padding[base+j+3]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[4] = log(1 + tf_padding[base+j+4]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[5] = log(1 + tf_padding[base+j+5]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[6] = log(1 + tf_padding[base+j+6]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_array[7] = log(1 + tf_padding[base+j+7]/(MU * (cf[topics[n][11]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i+pos] + MU));
          score_vec = _mm256_load_ps((__m256 *)&score_array[0]);
          score_vec = _mm256_and_ps(score_vec, (__m256)mask);
          t1 = _mm256_hadd_ps(score_vec,score_vec);
          t2 = _mm256_hadd_ps(t1,t1);
          t3 = _mm256_extractf128_ps(t2,1);
          t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
          scores[pos] = scores[pos] + _mm_cvtss_f32(t4);
        }
      }
      
      if (unlikely(scores[0] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = scores[0];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[0] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = scores[0];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[1] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+1;
          float *scorez = malloc(sizeof(float)); *scorez = scores[1];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[1] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+1;
            float *scorez = malloc(sizeof(float)); *scorez = scores[1];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[2] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+2;
          float *scorez = malloc(sizeof(float)); *scorez = scores[2];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[2] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+2;
            float *scorez = malloc(sizeof(float)); *scorez = scores[2];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[3] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+3;
          float *scorez = malloc(sizeof(float)); *scorez = scores[3];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[3] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+3;
            float *scorez = malloc(sizeof(float)); *scorez = scores[3];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[4] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+4;
          float *scorez = malloc(sizeof(float)); *scorez = scores[4];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[4] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+4;
            float *scorez = malloc(sizeof(float)); *scorez = scores[4];
            heap_insert(h, scorez, docid);
          }
        }
      }
      if (unlikely(scores[5] > 0)) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i+5;
          float *scorez = malloc(sizeof(float)); *scorez = scores[5];
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (scores[5] > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i+5;
            float *scorez = malloc(sizeof(float)); *scorez = scores[5];
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += indexsum;
    }
  } else {
    for (i=start; i<end; i++) {
      score = 0;
      for (j=0; j<doclengths_ordered_padding[i]; j++) {
        for (t=2; t<2+topics[n][1]; t++) {
          if (unlikely(collection_tf_padding[base+j] == topics[n][t])) {
            score+=log(1 + tf_padding[base+j]/(MU * (cf[topics[n][t]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i] + MU));
          }
        }
      }
      
      if (score > 0) {
        int size = heap_size(h);
        
        if ( size < TOP_K ) {
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = score;
          heap_insert(h, scorez, docid);
        } else {
          heap_min(h, (void**)&min_key, (void**)&min_val);
          
          if (score > *min_key) {
            heap_delmin(h, (void**)&min_key, (void**)&min_val);
            
            int *docid = malloc(sizeof(int)); *docid = i;
            float *scorez = malloc(sizeof(float)); *scorez = score;
            heap_insert(h, scorez, docid);
          }
        }
      }
      
      base += doclengths_ordered_padding[i];
    }
  }
  
  for (; i<end; i++) {
    if (tweetids[i] > topics_time[n]) {
      base += doclengths_ordered_padding[i];
      continue;
    }
    score = 0;
    for (j=0; j<doclengths_ordered_padding[i]; j++) {
      for (t=2; t<2+topics[n][1]; t++) {
        if (unlikely(collection_tf_padding[base+j] == topics[n][t])) {
          score += log(1 + tf_padding[base+j]/(MU * (cf[collection_tf_padding[base+j]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i] + MU));
        }
      }
    }
    
    if (score > 0) {
      int size = heap_size(&h);
      
      if ( size < TOP_K ) {
        int *docid = malloc(sizeof(int)); *docid = i;
        float *scorez = malloc(sizeof(float)); *scorez = score;
        heap_insert(&h, scorez, docid);
      } else {
        heap_min(&h, (void**)&min_key, (void**)&min_val);
        
        if (score > *min_key) {
          heap_delmin(&h, (void**)&min_key, (void**)&min_val);
          
          int *docid = malloc(sizeof(int)); *docid = i;
          float *scorez = malloc(sizeof(float)); *scorez = score;
          heap_insert(&h, scorez, docid);
        }
      }
    }
    
    base += doclengths_ordered_padding[i];
  }
  arg->done=1;
  return 0;
}

