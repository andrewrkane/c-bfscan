
#include "include/threadpool.c"

int search(struct threadpool *pool, int nthreads, int n) {
  heap h_array[nthreads];
  memset(h_array,0,sizeof(h_array));
  int i=0;
  struct arg_struct *arglist = malloc(nthreads * sizeof (struct arg_struct));
  for (i=0; i<nthreads; i++) {
    struct arg_struct *args = &arglist[i];
    args->topic = n;
    args->startidx = i*(int)(ceil((double)num_docs / nthreads));
    if ((i+1)*(int)(ceil((double)num_docs / nthreads)) > num_docs) {
      args->endidx = num_docs;
    } else {
      args->endidx = (i+1)*(int)(ceil((double)num_docs / nthreads));
    }
    args->base = termindexes[nthreads-1][i];
    heap h;
    heap_create(&h,0,NULL);
    h_array[i] = h;
    args->h = &h_array[i];
    args->done=0;
    threadpool_add_task(pool,scansearch,args,0);
  }
  
  heap h_merge;
  heap_create(&h_merge,0,NULL);
  float* min_key_merge;
  int* min_val_merge;
  for (i=0; i<nthreads; i++) {
    struct arg_struct *args = &arglist[i];
    if (!args->done) { threadpool_wait_for_workers(pool, &args->done); }
    if (!args->done) { printf("ERROR: threadpool workers did not finish."); } // verify threadpool wait
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
    printf("MB%02d Q0 %ld %d %f " SCANNAME "_multithread_intraquery\n", (n+1), tweetids[*min_val_merge], rank, *min_key_merge);
    rank--;
  }
  heap_destroy(&h_merge);
  
  free(arglist); arglist=0;
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

    struct threadpool *pool;
    pool = threadpool_init(nthreads);

    int n;
    for (n=0; n<num_topics; n++) {
      search(pool, nthreads, n);
    }
    
    threadpool_free(pool,1);

    gettimeofday(&end, NULL);

    time_spent = (double)((end.tv_sec * 1000000 + end.tv_usec) - (begin.tv_sec * 1000000 + begin.tv_usec));
    total = total + time_spent / 1000.0;
  }
  printf("Total time = %f ms\n", total/N);
  printf("Time per query = %f ms\n", (total/N)/num_topics);
}
