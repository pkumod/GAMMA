NVCC=nvcc
NVCCFLAGS=-gencode arch=compute_70,code=sm_70 #--ptxas-options=-v
CXX=g++
CXXFLAGS=-O3 -std=c++11 #-fopemmp -Iinclude
DEBUGFLAGS=#-G -g
DEPENDENCY=accessMode.cuh graph.cuh utils.h embedding.cuh expand.cuh log.h clock.h
all: kcl fsm sm

sm : sm.cu ${DEPENDENCY} queryGraph.cuh log.o
	${NVCC} sm.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o sm -std=c++11
kcl : kcl.cu ${DEPENDENCY} log.o
	${NVCC} kcl.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o kcl -std=c++11
fsm : fsm.cu ${DEPENDENCY} aggregrate.cuh log.o
	${NVCC} fsm.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o fsm -std=c++11
multimerge: multimerge.cu ${DEPENDENCY} 
	${NVCC} multimerge.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o multimerge -std=c++11
log.o: log.cpp 
	$(CXX) $(CXXFLAGS) -c log.cpp -o log.o
clean:
	rm -f kcl *.o 
