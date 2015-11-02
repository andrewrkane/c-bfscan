QUERY = twitter-tools-core/query_e50_freq.h

TI=-include ~/Large/tweets/TermStats/termindexes.h
TIP=-include ~/Large/tweets/TermStats/termindexes_padding.h

GCC=gcc -O3 -w
GCCAVX=gcc -O3 -w -msse4.1 -mavx2

all: Scan1 Scan2 AVXScan1 AVXScan2


Scan1: Scan1.h Scan*.c include/*.h include/*.c
	$(GCC) Scan_singlethread.c -o Scan1_sg.exe -include $(QUERY) -include Scan1.h
	$(GCC) Scan_multithread_intraquery.c -o Scan1_ra.exe -include $(QUERY) -include Scan1.h $(TI)
	$(GCC) Scan_multithread_interquery.c -o Scan1_er.exe -include $(QUERY) -include Scan1.h $(TI)


Scan2: Scan2.h Scan*.c include/*.h include/*.c
	$(GCC) Scan_singlethread.c -o Scan2_sg.exe -include $(QUERY) -include Scan2.h
	$(GCC) Scan_multithread_intraquery.c -o Scan2_ra.exe -include $(QUERY) -include Scan2.h $(TI)
	$(GCC) Scan_multithread_interquery.c -o Scan2_er.exe -include $(QUERY) -include Scan2.h $(TI)


AVXScan1: AVXScan1*.c include/*.h include/*.c
	$(GCCAVX) AVXScan1.c -o AVXScan1_sg.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan1_multithread_intraquery.c -o AVXScan1_ra.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan1_multithread_interquery.c -o AVXScan1_er.exe -include $(QUERY) $(TIP)


AVXScan2: AVXScan2*.c include/*.h include/*.c
	$(GCCAVX) AVXScan2.c -o AVXScan2_sg.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan2_multithread_intraquery.c -o AVXScan2_ra.exe -include $(QUERY) $(TIP)
	$(GCCAVX) AVXScan2_multithread_interquery.c -o AVXScan2_er.exe -include $(QUERY) $(TIP)


clean:
	rm -f Scan1_*.exe Scan2*.exe AVXScan1*.exe AVXScan2*.exe


