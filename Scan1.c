#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "include/constants.h"
#include "include/data.c"
#include "include/heap.c"

#define COUNT_SCORING false

extern void init_tf(char * data_path);
int num_docs;
int total_terms;
int num_topics;
int main(int argc, const char* argv[]) {
  init_tf(argv[1]);

  int i=0;

  clock_t begin, end;
  double time_spent;
  begin = clock();

  int base = 0;
  float score=0;

  int n;
  int t;

#if COUNT_SCORING
  int scored=0, nonscored=0;
#endif
  for (n=0; n<num_topics; n++) {
    heap h;
    heap_create(&h,0,NULL);

    float* min_key;
    int* min_val;
    float min_score=0;

    int start_doc=0, end_doc=num_docs;
    if (tweetids[end_doc-1] > topics_time[n]) { end_doc--;
      for (;;) { int h=(start_doc+end_doc)/2; if (h==end_doc) break; if (tweetids[h] > topics_time[n]) end_doc=h; else start_doc=h+1; }
    }

    base = 0;
    for (i=0; i<end_doc; i++) {
      for (int base_end = base+doclengths_ordered[i]; base<base_end; base++) {
        for (t=0; t<topics[n][1]; t++) {
          if (collection_tf[base] == topics[n][t+2]) {
            score+=topicsfreq[n][t]*( log(1 + tf[base]/(MU * (cf[topics[n][t+2]] + 1) / (total_terms + 1))) + log(MU / (doclengths[i] + MU)) );
            break;
          }
        }
      }
#if COUNT_SCORING
      if (score>0) scored++; else nonscored++;
#endif
      if (score > 0) {
        // debugging
        //printf("score=%f\n",score);
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
      printf("MB%02d Q0 %ld %d %f Scan1\n", (n+1), tweetids[*min_val], rank, *min_key);
      rank--;
    }

    heap_destroy(&h);
  }
#if COUNT_SCORING
  printf("num_docs=%d, scored=%d, nonscored=%d\n",num_docs,scored,nonscored);
#endif

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total time = %f ms\n", time_spent * 1000);
  printf("Time per query = %f ms\n", (time_spent * 1000)/num_topics);
  printf("Throughput: %f qps\n", num_topics/time_spent);
}