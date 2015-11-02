
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
    printf("MB%02d Q0 %ld %d %f " SCANNAME "\n", (n+1), tweetids[*min_val], rank, *min_key);
    rank--;
  }
  
  heap_destroy(&h);
  free(args);
}

int main(int argc, const char* argv[]) {
  init_tf(argv[1]);

  clock_t begin, end;
  double time_spent;
  begin = clock();

  int n;
  for (n=0; n<num_topics; n++) {
    search(n);
  }

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total time = %f ms\n", time_spent * 1000);
  printf("Time per query = %f ms\n", (time_spent * 1000)/num_topics);
  printf("Throughput: %f qps\n", num_topics/time_spent);
}