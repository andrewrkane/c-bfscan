
#define unlikely(expr) __builtin_expect(!!(expr),0)
#define likely(expr) __builtin_expect(!!(expr),1)

struct arg_struct {
    int topic;
    int startidx;
    int endidx;
    int base;
    heap* h;
    int done;
};

#define PREFETCHC //__builtin_prefetch(&collection_tf[base+1024]);

int scansearch(struct arg_struct *arg) {
  int n = arg->topic;
  int start = arg->startidx;
  int end = arg->endidx;
  heap* h = arg->h;

  int i=0;
  int base = arg->base;
  float score=0;
  int t;
  float* min_key;
  int* min_val;
  float min_score=0;

  int low=start, high=end;
  if (tweetids[high-1] > topics_time[n]) { high--;
    for (;;) { int p=(low+high)/2; if (p==high) break; if (tweetids[p] > topics_time[n]) high=p; else low=p+1; }
  }

  for (i=start; likely(i<high); i++) {
    PREFETCHC
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
      score = 0;
    }
  }
  arg->done=1;
  return 0;
}

