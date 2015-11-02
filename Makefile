QUERY = twitter-tools-core/query_e50_freq.h

TI=-include ~/Large/tweets/TermStats/termindexes.h
TIP=-include ~/Large/tweets/TermStats/termindexes_padding.h

GCC=gcc -O3 -w
GCCAVX=gcc -O3 -w -msse4.1 -mavx2

all: Scan1 Scan2 AVXScan1 AVXScan2


Scan1: Scan1*.c include/*.h include/*.c
	$(GCC) Scan1.c -o Scan1_single.exe -include $(QUERY)
	$(GCC) Scan1_multithread_intraquery.c -o Scan1_intra.exe -include $(QUERY) $(TI)
	$(GCC) Scan1_multithread_interquery.c -o Scan1_inter.exe -include $(QUERY) $(TI)


Scan2: Scan2*.c include/*.h include/*.c
	$(GCC) Scan2.c -o Scan2_single.exe -include $(QUERY)
	$(GCC) Scan2_multithread_intraquery.c -o Scan2_intra.exe -include $(QUERY) $(TI)
	$(GCC) Scan2_multithread_interquery.c -o Scan2_inter.exe -include $(QUERY) $(TI)


AVXScan1: AVXScan1*.c include/*.h include/*.c
	$(GCCAVX) AVXScan1.c -o AVXScan1_single.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan1_multithread_intraquery.c -o AVXScan1_intra.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan1_multithread_interquery.c -o AVXScan1_inter.exe -include $(QUERY) $(TIP)


AVXScan2: AVXScan2*.c include/*.h include/*.c
	$(GCCAVX) AVXScan2.c -o AVXScan2_single.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan2_multithread_intraquery.c -o AVXScan2_intra.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan2_multithread_interquery.c -o AVXScan2_inter.exe -include $(QUERY) $(TIP)


clean:
	rm Scan1_*.exe Scan2*.exe AVXScan1*.exe AVXScan2*.exe


