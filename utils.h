#ifndef UTILS_H
#define UTILS_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <set>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cassert>
#include <fstream>
#include <fcntl.h>
#include <cassert>
#include <unistd.h>
#include <stdint.h>
#include <algorithm>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include "log.h"
#include "clock.h"

typedef uint32_t KeyT;
typedef uint64_t OffsetT;
typedef uint32_t index_type;
typedef uint8_t edge_data_type;
typedef uint8_t node_data_type;
typedef uint8_t key_type;
typedef uint8_t history_type;
typedef uint8_t label_type;
typedef uint64_t emb_off_type;
typedef unsigned char SetType;
typedef unsigned long long AccType;
typedef uint32_t ATT;
#define embedding_max_length 7
#define BLOCK_SIZE 256
#define max_label 1024
#define expand_batch_size (1<<24)//20000000
#define EMB_FTR_CACHE_SIZE 600000000 
#define FIRST(x) ((x>>32)&0xffffffff)
#define SECOND(x) (x&0xffffffff)
#define MAX_EMB_UNIT_NUM 6000000000
static void check_cuda(const cudaError_t e, const char* file,
                             const int line) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
    exit(1);
  }
}
#define check_cuda_error(x) check_cuda(x, __FILE__, __LINE__)

inline __device__ void swap(uint64_t &e) {
	int first = FIRST(e);
	int second = SECOND(e);
	if (first > second) {
		e = (((uint64_t)first)<<32)|second; 
	}
}
inline __device__ int compare_edge(uint64_t e1, uint64_t e2) {
	swap(e1);
	swap(e2);
	if (FIRST(e1) == FIRST(e2))
		return SECOND(e1) - SECOND(e2);
	else 
		return FIRST(e1) - FIRST(e2);
}

typedef enum{
	GPU_MEM = 0,
	UNIFIED_MEM = 1,
	ZERO_COPY_MEM = 2,
	COMBINED_MEM = 3,
} mem_type;
template <typename KeyType>
__host__ __device__ inline int32_t binarySearch(KeyType *list, uint32_t list_length, KeyType value) {
	uint32_t s = 0, e = list_length;
	while (s < e) {
		uint32_t mid = (s + e)/2;
		KeyType tmp_value =  list[mid];
		if (tmp_value == value)
			return mid;
		else if (tmp_value < value) 
			s = mid + 1;
		else 
			e = mid;
	}
	return -1;
}
struct is_valid{
	__host__ __device__ bool operator()(const uint8_t &x) {
		return x == 1;
	}
};
struct patternID {
	uint32_t nbr;
	uint64_t lab;
	__device__ __host__ patternID():nbr(), lab() {}
	__device__ __host__ patternID(uint32_t n, uint64_t l): nbr(n), lab(l) {}
	__device__ __host__ bool operator < (const patternID& p) {
		return nbr < p.nbr || (nbr == p.nbr && lab < p.lab);
	}
	__device__ __host__ bool operator == (const patternID& p) {
		return nbr ==  p.nbr &&  lab == p.lab;
	}
	__device__ __host__ patternID & operator = (const patternID& p) {
		if (this != &p) {
			nbr = p.nbr;
			lab = p.lab;
		}
		return *this;
	}
};
#endif
