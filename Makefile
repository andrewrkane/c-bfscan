QUERY = twitter-tools-core/query_e50_freq.h

TI=-include ~/Large/tweets/TermStats/termindexes.h
TIP=-include ~/Large/tweets/TermStats/termindexes_padding.h

GCC=gcc -std=gnu99 -O3 -w
GCCAVX=gcc -std=gnu99 -O3 -w -msse4.1 -mavx2
LIB=-lm -lpthread

all: Scan1 Scan2 AVXScan1 AVXScan2


Scan1: Scan1.h Scan*.c include/*.h include/*.c
	$(GCC) Scan_singlethread.c -o Scan1_sg.exe -include $(QUERY) -include Scan1.h $(LIB)
	$(GCC) Scan_multithread_intraquery.c -o Scan1_ra.exe -include $(QUERY) -include Scan1.h $(TI) $(LIB)
	$(GCC) Scan_multithread_interquery.c -o Scan1_er.exe -include $(QUERY) -include Scan1.h $(TI) $(LIB)


Scan2: Scan2.h Scan*.c include/*.h include/*.c
	$(GCC) Scan_singlethread.c -o Scan2_sg.exe -include $(QUERY) -include Scan2.h $(LIB)
	$(GCC) Scan_multithread_intraquery.c -o Scan2_ra.exe -include $(QUERY) -include Scan2.h $(TI) $(LIB)
	$(GCC) Scan_multithread_interquery.c -o Scan2_er.exe -include $(QUERY) -include Scan2.h $(TI) $(LIB)


AVXScan1: AVXScan1.h Scan*.c include/*.h include/*.c
	$(GCCAVX) Scan_singlethread.c -o AVXScan1_sg.exe -include $(QUERY) -include AVXScan1.h $(TIP) $(LIB)
	$(GCCAVX) Scan_multithread_intraquery.c -o AVXScan1_ra.exe -include $(QUERY) -include AVXScan1.h $(TIP) $(LIB)
	$(GCCAVX) Scan_multithread_interquery.c -o AVXScan1_er.exe -include $(QUERY) -include AVXScan1.h $(TIP) $(LIB)

AVXScan2: AVXScan2.h Scan*.c include/*.h include/*.c
	$(GCCAVX) Scan_singlethread.c -o AVXScan2_sg.exe -include $(QUERY) -include AVXScan2.h $(TIP) $(LIB)
	$(GCCAVX) Scan_multithread_intraquery.c -o AVXScan2_ra.exe -include $(QUERY) -include AVXScan2.h $(TIP) $(LIB)
	$(GCCAVX) Scan_multithread_interquery.c -o AVXScan2_er.exe -include $(QUERY) -include AVXScan2.h $(TIP) $(LIB)


clean:
	rm -f Scan1_*.exe Scan2*.exe AVXScan1*.exe AVXScan2*.exe


