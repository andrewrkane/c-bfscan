
#include "include/threadpool.c"

int search(int n) {
  struct arg_struct *args = malloc(sizeof (struct arg_struct));
  args->topic = n;
  args->startidx = 0;
  args->endidx = num_docs;
  args->base = 0;
  heap h;
  heap_create(&h,0,NULL);
  args->h = &h;
  
  scansearch(args);
  
  float* min_key;
  int* min_val;
  int rank = TOP_K;
  while (heap_delmin(&h, (void**)&min_key, (void**)&min_val)) {
    printf("MB%02d Q0 %ld %d %f " SCANNAME "_multithread_interquery\n", (n+1), tweetids[*min_val], rank, *min_key);
    rank--;
  }
  
  heap_destroy(&h);
  free(args);
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
