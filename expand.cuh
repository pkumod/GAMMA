#ifndef EXPAND_H
#define EXPAND_H
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include "graph.cuh"
#include "embedding.cuh"
typedef enum{
	ID = 0,
	DEGREE = 1,
} emb_order;

struct expand_constraint{
public:
	expand_constraint(node_data_type _label, uint8_t degree_minimum, uint64_t _nbrs,
					  uint8_t _nbr_size, emb_order _emb_order_flag, //int8_t *_order_nbr_cmp,
					  uint64_t _order_nbr, uint8_t _order_nbr_size){
		assert(_nbr_size <= 8);
		assert(_order_nbr_size <= 8);
		label = _label;
		deg_minimum = degree_minimum;
		nbrs = _nbrs;
		nbr_size = _nbr_size;
		emb_order_flag = _emb_order_flag;
		//order_nbr_cmp = _order_nbr_cmp;
		order_nbr = _order_nbr;
		order_nbr_size = _order_nbr_size;
	}
	node_data_type label;
	uint8_t deg_minimum;
	uint64_t nbrs;//8 | 7 | 6 | 5 | 4 | 3 | 2 | 1
	uint8_t nbr_size;
	emb_order emb_order_flag;
	//int8_t* order_nbr_cmp;
	uint64_t order_nbr;
	uint8_t order_nbr_size;
};

inline __device__ void warp_reduce(uint32_t lane_id, uint32_t value, uint32_t &result) {
	result = value;
	uint32_t tmp_result;
	tmp_result = __shfl_up_sync(0xffffffff, result, 1, 32);
	if (lane_id >= 1) result += tmp_result;
	tmp_result = __shfl_up_sync(0xffffffff, result, 2, 32);
	if (lane_id >= 2) result += tmp_result;
	tmp_result = __shfl_up_sync(0xffffffff, result, 4, 32);
	if (lane_id >= 4) result += tmp_result;
	tmp_result = __shfl_up_sync(0xffffffff, result, 8, 32);
	if (lane_id >= 8) result += tmp_result;
	tmp_result = __shfl_up_sync(0xffffffff, result, 16, 32);
	if (lane_id >= 16) result += tmp_result;
	result -= value;
}
__device__ inline int32_t binary_search(KeyT *list, uint32_t list_length, KeyT value) {
	uint32_t s = 0, e = list_length;
	while (s < e) {
		uint32_t mid = (s + e)/2;
		KeyT tmp_value =  list[mid];
		if (tmp_value == value)
			return mid;
		else if (tmp_value < value) 
			s = mid + 1;
		else 
			e = mid;
	}
	return -1;
}
__device__ inline int32_t warp_binary_search(KeyT *list, uint32_t list_length, KeyT value,
											   uint32_t lane_id, KeyT *buffer) {
	buffer[lane_id] = list[list_length*lane_id/32];
	__syncwarp();
	int32_t bot = 0, top = 32, mid;
	while (top > bot + 1) {
		mid = (top + bot)/2;
		int32_t X = buffer[mid];
		if (value == X)
			return mid;
		else if (X < value)
			bot = mid;
		else
			top = mid;
	}
	bot = bot * list_length/32;
	top = top * list_length/32;
	while (bot < top) {
		mid = (bot + top)/2;
		int32_t X = list[mid];
		if (X == value)
			return mid;
		else if(X < value)
			bot = mid + 1;
		else 
			top = mid;
	}
	return -1;
}
__device__ inline bool emb_validation_check(KeyT *emb, CSRGraph g, int level, expand_constraint ec, 
											KeyT dst) {
	if(dst == 0xffffffff) return false;
	//remove duplicated match of a same vertex
	for (int i = 0; i <= level; i ++)
		if (emb[i] == dst)
			return false;
	//label check
	if (ec.label != 0xff && g.getData(dst) != ec.label)
		return false;
	//minimum degree check
	if (g.getDegree(dst) < ec.deg_minimum)
		return false;
	//emb order check
	uint64_t nbrs = ec.order_nbr;
	for (int i = 0; i < ec.order_nbr_size; i++) {
		switch (ec.emb_order_flag){
			case ID:
				if (dst < emb[(nbrs>>(8*i))&0xff]) return false;
				break;
			case DEGREE:
				uint32_t cur_nbr = emb[(nbrs>>(8*i))&0xff];
				uint32_t dst_deg = g.getDegree(dst);
				uint32_t nbr_deg = g.getDegree(cur_nbr);
				if (dst_deg < nbr_deg || (dst_deg == nbr_deg && dst < cur_nbr)) return false;
				//if (g.getDegree(dst) < g.getDegree(emb[(nbrs>>(8*i))&0xff])) return false;
				break;
		}
	}
	//adjacency check
	nbrs = ec.nbrs;
	for (int i = 0; i < ec.nbr_size; i ++) {
		uint8_t matched = (nbrs>>(8*i))&0xff;
		KeyT matched_vertex = emb[matched];
		OffsetT row_begin = g.edge_begin(matched_vertex);
		uint32_t degree = g.getDegree(matched_vertex);
		KeyT *adj_list = g.getAdjListofSrc(matched_vertex, row_begin);
		if(binary_search(adj_list, degree, dst) == -1) {
			return false;
		}
	}
	return true;
}
__device__ inline void pre_merge(CSRGraph g, expand_constraint ec, uint8_t min_neighbor, KeyT *sh_emb,
								 KeyT *sh_buffer, uint32_t &com_nbr_size, uint32_t lane_id) {
	//put the minimum-length adjacency list into shared memory for later intersection
	KeyT min_nbr_node = sh_emb[min_neighbor]; 
	KeyT *min_nbr = g.getAdjListofSrc(min_nbr_node, g.edge_begin(min_nbr_node));
	KeyT min_nbr_size = g.getDegree(min_nbr_node);
	uint32_t write_pos = 0;
	for (uint32_t i = lane_id; i < (min_nbr_size+31)/32*32; i += 32) {
		uint32_t valid_match = (i < min_nbr_size) ? 1 : 0;
		KeyT dst;
		if (valid_match) {
			dst = min_nbr[i];
			if (ec.label != 0xff && g.getData(dst) != ec.label) {
				valid_match = 0;
			} else if (g.getDegree(dst) < ec.deg_minimum) {
				valid_match = 0;
			}
		}
		uint32_t results = 0;
		warp_reduce(lane_id, valid_match, results);
		uint32_t valid_total = results + valid_match;
		valid_total = __shfl_sync(0xffffffff, valid_total, 31);
		if (valid_match == 1) {
			sh_buffer[write_pos + results] = dst;
		}
		write_pos += valid_total;
	}
	__syncwarp();
	//multiple list intersection
	com_nbr_size = write_pos;
	for (uint32_t i = 0; i < ec.nbr_size-1; i ++) {
		uint8_t query_node = (ec.nbrs>>(8*i))&0xff;
		if (query_node == min_nbr_node)
			continue;
		uint32_t data_node = sh_emb[query_node];
		uint32_t deg = g.getDegree(data_node);
		KeyT *adj = g.getAdjListofSrc(data_node, g.edge_begin(data_node));
		write_pos = 0;
		for (uint32_t j = lane_id; j < (com_nbr_size+31)/32*32; j += 32) {
			uint32_t _key = (j < com_nbr_size) ? sh_buffer[j] : 0xffffffff;
			uint32_t matched = 0;
			//__syncwarp();
			if (binary_search(adj, deg, _key) != -1) {
				matched = 1;
			}
			uint32_t results = 0;
			warp_reduce(lane_id, matched, results);
			uint32_t valid_total = results + matched;
			valid_total = __shfl_sync(0xffffffff, valid_total, 31);
			if (matched == 1)
				sh_buffer[write_pos+results] = _key;
			write_pos += valid_total;
			//__syncwarp();
		}
		com_nbr_size = write_pos;
	}
	return ;
}
//TODO we have not use COALESCED and ALIGNED memory access yet
//TODO parameter control to save more registers for the program
//TODO the register split can be detected using nvprof with flag  --ptx?????
__global__ void extend_alloc(EmbeddingList emb, int level, CSRGraph g,  
							 expand_constraint ec, uint32_t* emb_row_off, 
							 emb_off_type base_off, uint32_t f_size) {
	//TODO: here we use a warp to deal with an embedding, later we can try dynamic work distribution methods:
	//while the workload of an embedding is heavy, use a block instead. (WARP VOTE MECHANISM)
	uint32_t total_warp = (blockDim.x*gridDim.x)>>5;
	__shared__ KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
	for (uint32_t _i = (threadIdx.x + blockIdx.x * blockDim.x)>>5; _i < f_size; _i += total_warp) {
		emb_off_type i = base_off + _i;
		emb.get_embedding(level, i, sh_emb[threadIdx.x]);//TODO same work for a warp
		//KeyT vid = emb_list.get_vid(level, i);
		KeyT vid = sh_emb[threadIdx.x][level];
		if (vid == 0xffffffff)
		    continue;
		uint32_t local_count = 0;
		//__threadfence_block();
		OffsetT row_begin = g.edge_begin(vid);
		uint32_t adj_size = g.edge_end(vid) - row_begin;
		//TODO for now we have not carry out the adjacency check, and we plan to use this constraint to reduce the candidate size of emb_validation_check in the future. If the minimum list length of several lists to be merged is less than a threshold, we merge all the adjacency lists at first to get L0,and then intetersect it with the below list adj, to get final results. For now, we use the naive methods: intersect the adj with several other adjacency list respectively.
		//TODO here for the expand process we only consider expanding from the last element in emb 
		KeyT *adj = g.getAdjListofSrc(vid, row_begin);
		for (uint32_t e = threadIdx.x&31; e < adj_size; e += 32) {
			KeyT dst = adj[e];
			if (emb_validation_check(sh_emb[threadIdx.x], g, level, ec, dst))
				local_count += 1;
		}
		local_count += __shfl_down_sync(0xffffffff, local_count, 16);
		local_count += __shfl_down_sync(0xffffffff, local_count, 8);
		local_count += __shfl_down_sync(0xffffffff, local_count, 4);
		local_count += __shfl_down_sync(0xffffffff, local_count, 2);
		local_count += __shfl_down_sync(0xffffffff, local_count, 1);
		if (threadIdx.x%32 == 0) {
			//if (local_count != adj_size)
			//	printf("local_count %d adj_size %d\n", local_count, adj_size);
			emb_row_off[_i] = local_count;
		}
	} 
	return ;
}
__global__ void extend_insert_indevice(EmbeddingList emb, int level, CSRGraph g,  
							 		   expand_constraint ec, uint32_t* emb_row_off, 
							 		   emb_off_type read_base_off, 
									   uint32_t f_size, KeyT *vid_cache, emb_off_type *idx_cache) {
	//TODO: here we use a warp to deal with an embedding, later we can try dynamic work distribution methods:
	//while the workload of an embedding is heavy, use a block instead. (WARP VOTE MECHANISM)
	uint32_t total_warp = (blockDim.x*gridDim.x)>>5;
	__shared__ KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
	__shared__ KeyT sh_base_off[BLOCK_SIZE/32];
	uint32_t lane_id = threadIdx.x%32;
	for (uint32_t _i = (threadIdx.x + blockIdx.x * blockDim.x)>>5; _i < f_size; _i += total_warp) {
		emb_off_type i = read_base_off + _i;
		emb.get_embedding(level, i, sh_emb[threadIdx.x]);
		KeyT vid = emb.get_vid(level, i);
		//KeyT vid = sh_emb[threadIdx.x][level];
		if (lane_id == 0) sh_base_off[threadIdx.x>>5] = emb_row_off[_i];
		__syncwarp();
		//__threadfence();
		if (vid == 0xffffffff)	
		    continue;
		OffsetT row_begin = g.edge_begin(vid);
		uint32_t adj_size = g.edge_end(vid) - row_begin;
		//TODO for now we have not carry out the adjacency check, and we plan to use this constraint to reduce the candidate size of emb_validation_check in the future. If the minimum list length of several lists to be merged is less than a threshold, we merge all the adjacency lists at first to get L0,and then intetersect it with the below list adj, to get final results. For now, we use the naive methods: intersect the adj with several other adjacency list respectively.
		//TODO here for the expand process we only consider expanding from the last element in emb 
		KeyT *adj = g.getAdjListofSrc(vid, row_begin);
		for (uint32_t e = threadIdx.x&31; e < (adj_size+31)/32*32; e += 32) {
			//WARNING : if this is ok if not all threads are active for warp reduce?
			KeyT dst = (e < adj_size) ? adj[e] : 0xffffffff;
			uint32_t result = 0, value = 0;
			uint32_t check_result = (emb_validation_check(sh_emb[threadIdx.x], g, level, ec, dst) == true)? 1: 0;
			value += check_result;
			warp_reduce(lane_id, value, result);
			if (check_result) {
				uint32_t pos = sh_base_off[threadIdx.x>>5] + result;
				vid_cache[pos] = dst;
				idx_cache[pos] = i;
			}
			if(lane_id == 31)
				sh_base_off[threadIdx.x>>5] += (result + value);
			__syncwarp();
			//__threadfence();
		}
	} 
	return ;
}
/*__global__ void extend_insert(EmbeddingList emb, int level, CSRGraph g,  
							 expand_constraint ec, uint32_t* emb_row_off, 
							 emb_off_type read_base_off, emb_off_type write_base_off, uint32_t f_size) {
	//TODO: here we use a warp to deal with an embedding, later we can try dynamic work distribution methods:
	//while the workload of an embedding is heavy, use a block instead. (WARP VOTE MECHANISM)
	uint32_t total_warp = (blockDim.x*gridDim.x)>>5;
	__shared__ KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
	__shared__ KeyT sh_base_off[BLOCK_SIZE/32];
	uint32_t lane_id = threadIdx.x%32;
	for (uint32_t _i = (threadIdx.x + blockIdx.x * blockDim.x)>>5; _i < f_size; _i += total_warp) {
		emb_off_type i = read_base_off + _i;
		emb.get_embedding(level, i, sh_emb[threadIdx.x]);
		KeyT vid = emb.get_vid(level, i);
		//KeyT vid = sh_emb[threadIdx.x][level];
		if (lane_id == 0) sh_base_off[threadIdx.x>>5] = write_base_off + emb_row_off[_i];
		__syncwarp();
		//__threadfence();
		OffsetT row_begin = g.edge_begin(vid);
		uint32_t adj_size = g.edge_end(vid) - row_begin;
		//TODO for now we have not carry out the adjacency check, and we plan to use this constraint to reduce the candidate size of emb_validation_check in the future. If the minimum list length of several lists to be merged is less than a threshold, we merge all the adjacency lists at first to get L0,and then intetersect it with the below list adj, to get final results. For now, we use the naive methods: intersect the adj with several other adjacency list respectively.
		//TODO here for the expand process we only consider expanding from the last element in emb 
		KeyT *adj = g.getAdjListofSrc(vid, row_begin);
		for (uint32_t e = threadIdx.x&31; e < (adj_size+31)/32*32; e += 32) {
			//WARNING : if this is ok if not all threads are active for warp reduce?
			KeyT dst = (e < adj_size) ? adj[e] : 0xffffffff;
			uint32_t result = 0, value = 0;
			uint32_t check_result = (emb_validation_check(sh_emb[threadIdx.x], g, level, ec, dst) == true)? 1: 0;
			value += check_result;
			warp_reduce(lane_id, value, result);
			if (check_result) {
				uint32_t pos = sh_base_off[threadIdx.x>>5] + result;
				emb.set_idx(level+1, pos, i);
				emb.set_vid(level+1, pos, dst);
			}
			if(lane_id == 31)
				sh_base_off[threadIdx.x>>5] += (result + value);
			__syncwarp();
			//__threadfence();
		}
	} 
	return ;
}*/
#define warp_max_nbr 1000
#define warp_write_chunk 128
#define warp_process_size 32
__global__ void extend_indevice(EmbeddingList emb, int level, CSRGraph g,  
							 		   expand_constraint ec, emb_off_type read_base_off, 
									   uint32_t f_size, KeyT *vid_cache, emb_off_type *idx_cache,
									   uint32_t *counter) {
	//TODO: here we use a warp to deal with an embedding, later we can try dynamic work distribution methods:
	//while the workload of an embedding is heavy, use a block instead. (WARP VOTE MECHANISM)
	uint32_t total_warp = (blockDim.x*gridDim.x)>>5;
	__shared__ KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
	uint32_t write_chunk_off = 0, inside_chunk_off = 0;
	uint32_t lane_id = threadIdx.x%32;
	if (lane_id == 0) 
		write_chunk_off = atomicAdd(counter, warp_write_chunk);
	write_chunk_off = __shfl_sync(0xffffffff, write_chunk_off,0);
	assert(write_chunk_off < EMB_FTR_CACHE_SIZE);
	for (uint32_t _i = (threadIdx.x + blockIdx.x * blockDim.x)>>5; _i < f_size; _i += total_warp) {
		emb_off_type i = read_base_off + _i;
		emb.get_embedding(level, i, sh_emb[threadIdx.x]);
		uint32_t emb_valid = 1;
		for (uint32_t j = 0; j <= level; j ++)
			if (sh_emb[threadIdx.x][j] == 0xffffffff) {
				emb_valid = 0;
				break;
			}
		if (emb_valid == 0) continue;
		//KeyT vid = emb.get_vid(level, i);
		KeyT vid = sh_emb[threadIdx.x][level];
		__syncwarp();
		//__threadfence();
		OffsetT row_begin = g.edge_begin(vid);
		uint32_t adj_size = g.edge_end(vid) - row_begin;
		//TODO here for the expand process we only consider expanding from the last element in emb 
		KeyT *adj = g.getAdjListofSrc(vid, row_begin);
		for (uint32_t e = threadIdx.x&31; e < (adj_size+31)/32*32; e += 32) {
			//WARNING : if this is ok if not all threads are active for warp reduce?
			KeyT dst = (e < adj_size) ? adj[e] : 0xffffffff;
			uint32_t result = 0, value = 0;
			uint32_t check_result = (emb_validation_check(sh_emb[threadIdx.x], g, level, ec, dst) == true)? 1: 0;
			value += check_result;
			warp_reduce(lane_id, value, result);
			uint32_t total_valid_num = value + result;
			total_valid_num = __shfl_sync(0xffffffff, total_valid_num, 31);
			//if (level == 0 && lane_id == 0 && total_valid_num != 32 && e + 32 < f_size)
			//	printf("the valid num is %d\n", total_valid_num);
			if (total_valid_num + inside_chunk_off >= warp_write_chunk) {
					for (uint32_t p = inside_chunk_off + lane_id; p < warp_write_chunk; p += 32) {
						idx_cache[write_chunk_off + p] = e;
						vid_cache[write_chunk_off + p] = 0xffffffff;
					}
					if (lane_id == 0) {
						write_chunk_off = atomicAdd(counter, warp_write_chunk);
					}
					write_chunk_off = __shfl_sync(0xffffffff, write_chunk_off, 0);
					inside_chunk_off = 0;
			}
			if(value == 1) {
					idx_cache[write_chunk_off + inside_chunk_off + result] = i;
					vid_cache[write_chunk_off + inside_chunk_off + result] = dst;
			}
			inside_chunk_off += total_valid_num;
			__syncwarp();
			//__threadfence();
		}
	} 
	return ;
}
__global__ void expand_kernel(EmbeddingList emb_list, int level, CSRGraph g, expand_constraint ec, emb_off_type base_off, 							   uint32_t f_size, KeyT *emb_vid, emb_off_type *emb_idx, uint32_t *counter) {
	uint32_t total_warp = (blockDim.x * gridDim.x)>>5;
	__shared__ KeyT sh_emb[BLOCK_SIZE/32][embedding_max_length];//sh_mem cache for warp-level embedding
	__shared__ KeyT sh_com_nbr[BLOCK_SIZE/32][warp_max_nbr];//sh_mem cache for pre-merged lists
	__shared__ uint8_t sh_emb_group[BLOCK_SIZE];//group embeddings according to their prefix for better intersection perf
	uint32_t warp_id = (threadIdx.x + blockDim.x*blockIdx.x)/32;
	uint32_t lane_id = threadIdx.x%32;
	uint32_t write_chunk_off = 0, inside_chunk_off = 0;
	//warps get their inital space for frontier expand
	if (lane_id == 0) {
		write_chunk_off = atomicAdd(counter, warp_write_chunk);
		assert(write_chunk_off < EMB_FTR_CACHE_SIZE);
	}
	__syncwarp();
	write_chunk_off = __shfl_sync(0xffffffff, write_chunk_off, 0);
	//allocate embeddeing intermediate results for warps
	//TODO: here we assume f_size is multiple of warp_process_size, and this can be implmented later in our codes.
	for (uint32_t _i = warp_id * warp_process_size; _i < f_size; _i += (total_warp * warp_process_size)) {
		//check embedding similarity and group them
		if (level == 0) sh_emb_group[threadIdx.x] = 1;
		else sh_emb_group[threadIdx.x] = 0xff;
		//here we use sh_com_nbr as temperal emb buffer to group them
		KeyT *local_emb = sh_com_nbr[threadIdx.x/32] + embedding_max_length*lane_id;
		emb_list.get_embedding(level, base_off+_i+lane_id, local_emb);
		for (uint32_t j = 0; j < ec.nbr_size-1; j ++) {//TODO:here we only consider warp_process_size = 32;
			uint8_t query_node = (ec.nbrs>>(8*j))&0xff;
			KeyT vid = local_emb[query_node];
			//KeyT vid = emb_list.d_vid_lists[query_node][vid_off];
			KeyT up_vid = __shfl_up_sync(0xffffffff, vid, 1);
			if (vid != up_vid)
				sh_emb_group[threadIdx.x] = 1;
		}//set flags
		__threadfence_block();
		if (lane_id == 0) {
			sh_emb_group[threadIdx.x] = 1;
			uint32_t write_pos = 0;
			for (uint8_t j = 0; j < 32; j ++) {
				if (sh_emb_group[threadIdx.x + j] == 1) {
					sh_emb_group[threadIdx.x + write_pos] = j;
					write_pos ++;
				}
			}
			if (write_pos < 32) sh_emb_group[threadIdx.x + write_pos] = 32;
		}//set embedding group as [begin1, begin2, begin3 ... 32)
		__threadfence_block();
		//for each group of embedding, pre-merge concerned adjacency list
		uint32_t group_id = 0;
		uint32_t first_warp_id = threadIdx.x/32*32;
		while (group_id < 32 && sh_emb_group[first_warp_id + group_id] < 32) {
			emb_off_type emb_start_idx = base_off + _i + sh_emb_group[first_warp_id + group_id];
			emb_off_type emb_end_idx = group_id >= 31 ? base_off + _i + 32 : 
										base_off + _i + sh_emb_group[first_warp_id + group_id + 1];
			if(lane_id == 0)
				emb_list.get_embedding(level, emb_start_idx, sh_emb[threadIdx.x/32]);
			__threadfence_block();
			//check whether the current group of embedding is a valid embedding
			bool valid_emb = true;
			for (uint32_t e = 0; e <= level; e ++)//TODO level > 0 ? level - 1 : 0
				if (sh_emb[threadIdx.x/32][e] == 0xffffffff)
					valid_emb = false;
			if (valid_emb == false) {
				group_id ++;
				continue;
			}
			//find the minimum length of several lists to be pre-merged
			uint8_t min_nbr_node = ec.nbrs&0xff;
			uint32_t min_nbr_size = g.getDegree(sh_emb[threadIdx.x/32][min_nbr_node]);
			for (uint32_t j = 0; j < ec.nbr_size-1; j ++) {//get minimum nbr list size
				uint8_t nbr_now = (ec.nbrs>>(8*j))&0xff;
				uint32_t nbr_size = g.getDegree(sh_emb[threadIdx.x/32][nbr_now]);
				if (nbr_size < min_nbr_size) {
					min_nbr_size = nbr_size;
					min_nbr_node = nbr_now;
				}
			}
			if (min_nbr_size >= warp_max_nbr || level == 0) {//TODO: this may cause severe workload imbalance 
				for (emb_off_type e = emb_start_idx; e < emb_end_idx; e ++) {
					KeyT emb_last = emb_list.get_vid(level, e);
					if (emb_last == 0xffffffff) 
						continue;
					/*if (level == 0) {//TODO: not necessary, better initialization methods instead
						KeyT v = sh_emb[threadIdx.x/32][0];
						if (ec.label != 0xff && g.getData(v) != ec.label) 
							continue;
						if (g.getDegree(v) < ec.deg_minimum)
							continue;
					}*/	
					uint32_t deg = g.getDegree(emb_last);
					KeyT *adj = g.getAdjListofSrc(emb_last, g.edge_begin(emb_last));
					for (uint32_t j = lane_id; j < (deg+31)/32*32; j += 32) {
						KeyT v = (j < deg) ? adj[j] : 0xffffffff;
						uint32_t matched = 1;
						//if (v == emb_last) matched = 0;
						if (v == 0xffffffff) 
							matched = 0;
						else {
							if (ec.label != 0xff && g.getData(v) != ec.label) {
								matched = 0;
							} else if (g.getDegree(v) < ec.deg_minimum) {
								matched = 0;
							}
							for (uint32_t i = 0; i < ec.nbr_size-1; i ++)
								if (v == sh_emb[threadIdx.x/32][i])
									matched = 0;
							uint64_t nbrs = ec.order_nbr;
							for (int i = 0; i < ec.order_nbr_size; i++) {
								uint8_t q_n = (nbrs>>(8*i))&0xff;
								uint32_t d_n;
								if (q_n == level)
									d_n = emb_last;
								else
									d_n = sh_emb[threadIdx.x/32][q_n];
								switch (ec.emb_order_flag){
									case ID:
										if (v < d_n) matched = 0;
										break;
									case DEGREE:
										uint32_t dst_deg = g.getDegree(v);
										uint32_t nbr_deg = g.getDegree(d_n);
										if (dst_deg < nbr_deg || (dst_deg == nbr_deg && v < d_n)) matched = 0;
										//if (g.getDegree(dst) < g.getDegree(emb[(nbrs>>(8*i))&0xff])) return false;
										break;
								}
							}
							for (uint32_t n = 0; n < ec.nbr_size-1; n ++) {
								uint8_t q_n = (ec.nbrs>>(8*n))&0xff;
								uint32_t d_n = sh_emb[threadIdx.x/32][q_n];
								uint32_t _deg = g.getDegree(d_n);
								KeyT *_adj = g.getAdjListofSrc(d_n, g.edge_begin(d_n));
								if (binary_search(_adj, _deg, v) == -1) {
									matched = 0;
									break;
								}
							}
						}
						__syncwarp();
						//matched = 1;
						uint32_t results = 0;
						warp_reduce(lane_id, matched, results);
						uint32_t total_valid_num = matched + results;
						total_valid_num = __shfl_sync(0xffffffff, total_valid_num, 31);
						//if (level == 0 && lane_id == 0 && total_valid_num != 32 && e + 32 < f_size)
						//	printf("the valid num is %d\n", total_valid_num);
						if (total_valid_num + inside_chunk_off >= warp_write_chunk) {
							for (uint32_t p = inside_chunk_off + lane_id; p < warp_write_chunk; p += 32) {
								emb_idx[write_chunk_off + p] = e;
								emb_vid[write_chunk_off + p] = 0xffffffff;
							}
							if (lane_id == 0) {
								write_chunk_off = atomicAdd(counter, warp_write_chunk);
							}
							write_chunk_off = __shfl_sync(0xffffffff, write_chunk_off, 0);
							inside_chunk_off = 0;
						}
						if(matched == 1) {
							emb_idx[write_chunk_off + inside_chunk_off + results] = e;
							emb_vid[write_chunk_off + inside_chunk_off + results] = v;
						}
						inside_chunk_off += total_valid_num;
					}
				}
			} else {//here we use pre-merge and two-way merge methods;
				uint32_t com_nbr_size = 0;
				pre_merge(g, ec, min_nbr_node, sh_emb[threadIdx.x/32], sh_com_nbr[threadIdx.x/32], com_nbr_size,lane_id);
				//merge final list with pre-merged lists and write new frontiers
				for (emb_off_type e = emb_start_idx; e < emb_end_idx; e ++) {
					KeyT emb_last = emb_list.get_vid(level, e);
					if (emb_last == 0xffffffff)
						continue;
					uint32_t deg = g.getDegree(emb_last);
					KeyT *adj = g.getAdjListofSrc(emb_last, g.edge_begin(emb_last));
					for (uint32_t j = lane_id; j < (deg+31)/32*32; j += 32) {
						KeyT v = (j < deg) ? adj[j] : 0xffffffff;
						uint32_t matched = 1;//(binary_search(sh_com_nbr[threadIdx.x/32], com_nbr_size, v) == -1) ? 0 : 1;
						//if (v == emb_last) matched = 0;
						if (v == 0xffffffff) 
							matched = 0;
						else {
							for (uint32_t i = 0; i < ec.nbr_size-1; i ++)
								if (v == sh_emb[threadIdx.x/32][i])
									matched = 0;
							uint64_t nbrs = ec.order_nbr;
							for (int i = 0; i < ec.order_nbr_size; i++) {
								uint8_t q_n = (nbrs>>(8*i))&0xff;
								uint32_t d_n;
								if (q_n == level)
									d_n = emb_last;
								else
									d_n = sh_emb[threadIdx.x/32][q_n];
								switch (ec.emb_order_flag){
									case ID:
										if (v < d_n) matched = 0;
										break;
									case DEGREE:
										uint32_t dst_deg = g.getDegree(v);
										uint32_t nbr_deg = g.getDegree(d_n);
										if (dst_deg < nbr_deg || (dst_deg == nbr_deg && v < d_n)) matched = 0;
										break;
								}
							}
							if(binary_search(sh_com_nbr[threadIdx.x/32], com_nbr_size, v) == -1)
								matched = 0;
						}
						uint32_t results = 0;
						warp_reduce(lane_id, matched, results);
						uint32_t total_valid_num = matched + results;
						total_valid_num = __shfl_sync(0xffffffff, total_valid_num, 31);
						if (total_valid_num + inside_chunk_off >= warp_write_chunk) {
							for (uint32_t p = inside_chunk_off + lane_id; p < warp_write_chunk; p += 32) {
								emb_idx[write_chunk_off + p] = e;
								emb_vid[write_chunk_off + p] = 0xffffffff;
							}
							if (lane_id == 0) {
								write_chunk_off = atomicAdd(counter, warp_write_chunk);
							}
							write_chunk_off = __shfl_sync(0xffffffff, write_chunk_off, 0);
							inside_chunk_off = 0;
						}
						if(matched == 1) {
							emb_idx[write_chunk_off + inside_chunk_off + results] = e;
							emb_vid[write_chunk_off + inside_chunk_off + results] = v;
						}
						inside_chunk_off += total_valid_num;
					}
				}
			}
			group_id ++;
		}

	}
	return ;
}
/*__global__ void emblist_check(EmbeddingList emb_list, uint32_t level, uint32_t emb_size) {
		uint32_t idx = threadIdx.x + blockIdx.x*blockDim.x;
		KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
		for (uint32_t i = idx>>5; i < emb_size; i += (gridDim.x*blockDim.x)/32) {
			emb_list.get_embedding(level, i, sh_emb[threadIdx.x]);
			if (i%(emb_size/8) == 0&& threadIdx.x%32 == 0)
				printf("bbb %d %d %d\n",sh_emb[threadIdx.x][0], sh_emb[threadIdx.x][1], sh_emb[threadIdx.x][2]); 
		}
}
void expand(CSRGraph &g, EmbeddingList &emb_list, int cur_level, expand_constraint &ec) {

	Clock exp("Inside Expand");
	exp.start();
	emb_off_type last_level_size = emb_list.size(cur_level-1);
	uint32_t *emb_row_off;
	uint32_t nblocks = 10000, nthreads = BLOCK_SIZE;//TODO tune them
	check_cuda_error(cudaMalloc((void **)&emb_row_off, sizeof(uint32_t)*(last_level_size+1)));
	check_cuda_error(cudaMemset(emb_row_off, 0, sizeof(uint32_t)*(last_level_size+1)));
	//TODO: for now we use alloc-insert methods, perhaps we can also try GSI's method:
	//preallocate maximum possible space for list intersection, and them compact, which save one 
	//expand time.
	log_info(exp.count("start extend alloc......"));
	extend_alloc<<<nblocks, nthreads>>>(emb_list, cur_level-1, g, ec, emb_row_off, 0, last_level_size);
	check_cuda_error(cudaDeviceSynchronize());
	thrust::exclusive_scan(thrust::device, emb_row_off, emb_row_off+last_level_size+1, emb_row_off);
	uint32_t new_emb_size;
	check_cuda_error(cudaMemcpy(&new_emb_size, emb_row_off+last_level_size, sizeof(uint32_t),cudaMemcpyDeviceToHost));
	log_info(exp.count("the frontier size of level %d is %u", cur_level, new_emb_size));
	// check the emb_row_off
	//uint32_t *emb_row_off_h = new uint32_t [last_level_size+1];
	//check_cuda_error(cudaMemcpy(emb_row_off_h, emb_row_off, sizeof(uint32_t)*(last_level_size+1), cudaMemcpyDeviceToHost));
	//for (uint32_t i = 0; i < last_level_size+1; i ++)
	//	if (i%(last_level_size/100)==0)
	//	printf("%d \n",emb_row_off_h[i]);
	//delete [] emb_row_off_h;
	
	emb_list.add_level(new_emb_size, cur_level);
	check_cuda_error(cudaDeviceSynchronize());
	//emblist_check<<<10000,nthreads>>>(emb_list, cur_level-1, last_level_size);
	log_info(exp.count("start extend insert......"));
	cudaDeviceSynchronize();
	extend_insert<<<nblocks, nthreads>>>(emb_list, cur_level-1, g, ec, emb_row_off, 0, 0, last_level_size); 
	check_cuda_error(cudaDeviceSynchronize());
	//printf("dasdefnwoicnoiwfyhl;oasdq\n");
	//emblist_check<<<10000,nthreads>>>(emb_list, cur_level, new_emb_size);
	check_cuda_error(cudaFree(emb_row_off));
	log_info(exp.count("end extend insert."));
	return ;
}*/
void expand_in_batch(CSRGraph &g, EmbeddingList & emb_list, int cur_level, expand_constraint &ec) {
	Clock exp("expand_in_batch");
	exp.start();
	emb_off_type last_level_size = emb_list.size(cur_level-1);
	uint32_t batch_num = (last_level_size + expand_batch_size-1)/expand_batch_size;
	emb_off_type *batch_expand_off = new emb_off_type [batch_num+1];
	uint32_t *emb_write_off = (uint32_t *)malloc(sizeof(uint32_t)*last_level_size);//TODO we use uint32_t instead of emb_off_type here
	memset(emb_write_off, 0, sizeof(uint32_t)*last_level_size);
	memset(batch_expand_off, 0, sizeof(emb_off_type)*(batch_num+1));
	uint32_t *emb_write_off_d;
	uint32_t nblocks = 10000;
	check_cuda_error(cudaMalloc((void **)&emb_write_off_d, sizeof(uint32_t)*(1+expand_batch_size)));
	//expand alloc
	for (uint32_t i = 0; i < batch_num; i ++) {
		check_cuda_error(cudaMemset(emb_write_off_d, 0, sizeof(uint32_t)*(1+expand_batch_size)));
		emb_off_type base_off = i * expand_batch_size;
		uint32_t cur_batch_size = (i < batch_num-1) ? expand_batch_size : (last_level_size - i*expand_batch_size);
		log_info("start extend malloc for chunk %d......", i);
		extend_alloc<<<nblocks, BLOCK_SIZE>>>(emb_list, cur_level-1, g, ec, emb_write_off_d, base_off, cur_batch_size);
		check_cuda_error(cudaDeviceSynchronize());
		thrust::exclusive_scan(thrust::device, emb_write_off_d, emb_write_off_d+cur_batch_size+1, emb_write_off_d);
		check_cuda_error(cudaMemcpy(emb_write_off+base_off, emb_write_off_d, sizeof(uint32_t)*cur_batch_size, cudaMemcpyDeviceToHost));
		check_cuda_error(cudaMemcpy(batch_expand_off+i, emb_write_off_d+cur_batch_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));//WARNING may lead to uint32_t copy to uint64_t
		check_cuda_error(cudaDeviceSynchronize());
		printf("for the %d th expand_alloc, the total size is %d\n", i, batch_expand_off[i]);
	}
	thrust::exclusive_scan(thrust::host, batch_expand_off, batch_expand_off+batch_num+1, batch_expand_off);
	//for (uint32_t i = 0; i <= batch_num; i ++)
	//	printf("%lu\n", batch_expand_off[i]);
	emb_off_type next_level_size = batch_expand_off[batch_num];
	emb_list.add_level(next_level_size, cur_level);
	KeyT *d_vid_cache;
	emb_off_type *d_idx_cache;
	uint32_t max_chunk_expand_size = 0;
	for (uint32_t i = 0; i < batch_num; i ++) {
		uint32_t expand_size = batch_expand_off[i+1]- batch_expand_off[i];
		if (expand_size > max_chunk_expand_size)
			max_chunk_expand_size = expand_size;
	}
	log_info("the max_chunk_expand_size is %u", max_chunk_expand_size);
	check_cuda_error(cudaMalloc((void **)&d_vid_cache, sizeof(KeyT)*max_chunk_expand_size));
	check_cuda_error(cudaMalloc((void **)&d_idx_cache, sizeof(emb_off_type)*max_chunk_expand_size));
	for (uint32_t i = 0; i < batch_num; i ++) {
		emb_off_type read_base_off = i*batch_num;
		emb_off_type write_base_off = batch_expand_off[i];
		uint32_t cur_batch_size = (i < batch_num -1)? expand_batch_size : (last_level_size - i*expand_batch_size);
		check_cuda_error(cudaMemcpy(emb_write_off_d, emb_write_off + read_base_off, sizeof(uint32_t)*cur_batch_size, cudaMemcpyHostToDevice));
		log_info("start extend insert for chuck %d... ...",i);
		extend_insert_indevice<<<nblocks, BLOCK_SIZE>>>(emb_list, cur_level-1, g, ec, emb_write_off_d,
											   			read_base_off, cur_batch_size, d_vid_cache, d_idx_cache);
		check_cuda_error(cudaDeviceSynchronize());
		uint32_t new_size = batch_expand_off[i+1] - batch_expand_off[i];
		//printf("new size %u, write base off %lu\n", new_size, write_base_off);
		emb_list.copy_to_vid(d_vid_cache, write_base_off, new_size, cur_level);
		emb_list.copy_to_idx(d_idx_cache, write_base_off, new_size, cur_level);
	}
	check_cuda_error(cudaFree(emb_write_off_d));
	check_cuda_error(cudaFree(d_idx_cache));
	check_cuda_error(cudaFree(d_vid_cache));
	delete [] batch_expand_off;
	free(emb_write_off);
	log_info("extend_alloc finished here, now we start extend_insert......");

}//in this expand function, the whole frontier are divided into batches, and then expanded seperately		
void expand_dynamic(CSRGraph &g, EmbeddingList &emb_list, int cur_level, expand_constraint &ec, bool copy_back) {
	Clock exp("expand in dynamic");
	exp.start();
	emb_off_type last_level_size = emb_list.size(cur_level-1);
	log_info("the last level size is %lu", last_level_size);
	uint32_t batch_num = (last_level_size + expand_batch_size-1)/expand_batch_size;
	log_info("the batch num is %d", batch_num);
	emb_off_type *batch_expand_off = new emb_off_type [batch_num+1];
	memset(batch_expand_off, 0, sizeof(emb_off_type)*(batch_num+1));
	uint32_t nblocks = 3000;
	KeyT *emb_vid_d;//maximum space preallocated in GPU, and all warps ask for space chucks in dynamic
	emb_off_type *emb_idx_d;//maximum space preallocated in GPU, and all warps ask for space chucks in dynamic
	check_cuda_error(cudaMalloc((void **)&emb_vid_d, sizeof(KeyT)*EMB_FTR_CACHE_SIZE));
	check_cuda_error(cudaMalloc((void **)&emb_idx_d, sizeof(emb_off_type)*EMB_FTR_CACHE_SIZE));
	uint32_t *global_counter;
	check_cuda_error(cudaMalloc((void **)&global_counter, sizeof(uint32_t)));
	emb_list.add_level(0, cur_level);
	emb_off_type valid_unit_num = 0;
	for (uint32_t i = 0; i < batch_num; i ++) {
		check_cuda_error(cudaMemset(global_counter, 0, sizeof(uint32_t)));
		check_cuda_error(cudaMemset(emb_vid_d, -1, sizeof(KeyT)*EMB_FTR_CACHE_SIZE));
		emb_off_type base_off = (emb_off_type)expand_batch_size*i;
		uint32_t cur_batch_size = (i < batch_num-1)? expand_batch_size:(last_level_size-i*expand_batch_size);
		log_info("start processing chunk %d",i);
		//extend_indevice<<<nblocks, BLOCK_SIZE>>>(emb_list, cur_level-1, g, ec, base_off, cur_batch_size,
		expand_kernel<<<nblocks, BLOCK_SIZE>>>(emb_list, cur_level-1, g, ec, base_off, cur_batch_size,
											   emb_vid_d, emb_idx_d, global_counter);
		check_cuda_error(cudaDeviceSynchronize());
		log_info("end kernel for chuck %d",i);
		check_cuda_error(cudaMemcpy(batch_expand_off+i, global_counter, sizeof(uint32_t),cudaMemcpyDeviceToHost));
		//assert(batch_expand_off[i] < EMB_FTR_CACHE_SIZE);
		valid_unit_num += emb_list.check_valid_num(emb_vid_d, batch_expand_off[i]);
		emb_off_type emb_size_now = emb_list.size(cur_level);
		if (copy_back) {
			emb_list.copy_to_vid_from_d(emb_vid_d, emb_size_now, batch_expand_off[i], cur_level);
			emb_list.copy_to_idx_from_d(emb_idx_d, emb_size_now, batch_expand_off[i], cur_level);
		}
		check_cuda_error(cudaDeviceSynchronize());
	}
	//TODO add emb-list adjustment here
	emb_list.size_adjustment();
	if (1) {
		uint64_t all_size = 0;
		all_size += g.nnodes * 8;
		all_size += g.nedges * 4 * 3;
		uint64_t embedTableSize = 0;
		for (int i = 0; i < emb_list.level(); i ++)
			embedTableSize += emb_list.size(i);
		embedTableSize += thrust::reduce(thrust::host, batch_expand_off, batch_expand_off+batch_num);
		embedTableSize *= (8+4+1);
		if (embedTableSize*0.8 > 5000000000) {
			embedTableSize += 5000000000;
		} else {
			embedTableSize *= 1.8;
		}
		all_size += embedTableSize;
		uint32_t *gc_h = new uint32_t [1];
		cudaMemcpy(gc_h, global_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		embedTableSize += gc_h[0] * 12;
		delete [] gc_h;
		printf("all used mem is %d MB, and data label is %d MB\n", all_size/1024/1024, g.nnodes/1024/1024);
	}
		
	check_cuda_error(cudaFree(emb_vid_d));
	check_cuda_error(cudaFree(emb_idx_d));
	check_cuda_error(cudaFree(global_counter));
	log_info(exp.count("end expand here, and valid emb number is %lu", valid_unit_num));
	delete [] batch_expand_off;
	return ;
}//in this expand function, the whole frontier are expanded in batches; what's more, we give local shared memory to each warp to cache local intersection results; lastly, we use dynamically methods to place the newly generated frontiers.








__global__ void check_emb_validation(EmbeddingList emblist, uint32_t level, uint8_t *valid_emb, uint32_t *counter, emb_off_type emb_size) {
		__shared__ KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
		uint32_t local_count = 0;
		for (emb_off_type i = threadIdx.x + blockDim.x*blockIdx.x; i < emb_size; i += blockDim.x*gridDim.x) {
			emblist.get_embedding(level, i, sh_emb[threadIdx.x]);
			bool valid = true;
			for (uint32_t j = 0; j <= level; j ++) {
				if (sh_emb[threadIdx.x][j] == 0xffffffff) {
					valid = false;
					break;
				}
			}
			if (valid) {
				local_count ++;
				valid_emb[i] = 1;
			}
		}
		local_count += __shfl_down_sync(0xffffffff, local_count, 16);
		local_count += __shfl_down_sync(0xffffffff, local_count, 8);
		local_count += __shfl_down_sync(0xffffffff, local_count, 4);
		local_count += __shfl_down_sync(0xffffffff, local_count, 2);
		local_count += __shfl_down_sync(0xffffffff, local_count, 1);
		if (threadIdx.x % 32 == 0)
			counter[(threadIdx.x+ blockDim.x*blockIdx.x)/32] = local_count;
		return ;
}
void emb_compaction(EmbeddingList emb_list, uint32_t l) {
		emb_off_type emb_size = emb_list.size(l);
		uint8_t *valid_emb;
		check_cuda_error(cudaMalloc((void **)&valid_emb, sizeof(uint8_t)*emb_size));
		check_cuda_error(cudaMemset(valid_emb, 0, sizeof(uint8_t)*emb_size));
		uint32_t block_num = 10000;
		uint32_t *counter;
		check_cuda_error(cudaMalloc((void **)&counter, sizeof(uint32_t)*BLOCK_SIZE/32*block_num));
		check_cuda_error(cudaMemset(counter, 0, sizeof(uint32_t)*BLOCK_SIZE/32*block_num));
		check_emb_validation<<<block_num, BLOCK_SIZE>>>(emb_list, l, valid_emb, counter, emb_size);
		check_cuda_error(cudaDeviceSynchronize());
		emb_off_type total_valid_emb = thrust::reduce(thrust::device, counter, counter + BLOCK_SIZE/32*block_num);
		check_cuda_error(cudaFree(counter));
		emb_list.compaction(l, valid_emb, total_valid_emb);
		check_cuda_error(cudaFree(valid_emb));
		return ;
}

__device__ inline bool quick_check(int size, KeyT *vids, label_type einfo2, label_type einfo, KeyT src, KeyT dst, CSRGraph g) {
	label_type v0l = g.getData(vids[0]);
	label_type dl = g.getData(dst);
	if (dl < v0l || (dl == v0l && dst <= vids[0]))	return false;
	if (size == 0)	return true;
	if (dst == vids[1]) return false;
	label_type v1l = g.getData(vids[1]);
	if (einfo == 0 && (dl < v1l || (dl == v1l && dst <= vids[1]))) return false;
	if (size == 1) {
	} else if (size == 2) {
		label_type v2l = g.getData(vids[2]);
		if (dst == vids[2]) return false;
		if (einfo == 0 && einfo2 == 0 && (dl < v2l || (dl == v2l && dst <= vids[2]))) return false;
		if (einfo == 1 && einfo2 == 1 && (dl < v2l || (dl == v2l && dst <= vids[2]))) return false;
	} else {
	}
	return true;
}
__device__ bool emb_check(KeyT *vid, label_type *einfo, int pos, int len, KeyT src, KeyT candi, CSRGraph g) {
	if(len < 3)
		return quick_check(len, vid, einfo[2], pos, src, candi, g);
	label_type cl = g.getData(candi);
	if (cl < g.getData(vid[0]) || (cl = g.getData(vid[0]) && candi <= vid[0]))//第一个节点必须最小
		return false;
	if (pos == 0 && (cl < g.getData(vid[1])||(cl == g.getData(vid[1]) &&candi <= vid[1])))//如果从第一个节点扩展，必须比已经扩展的兄弟节点大
		return false;
	if (candi == vid[einfo[pos]])//避免a->b->a
		return false;
	uint64_t added_edge = (((uint64_t)src) << 32) | candi;
	uint64_t added_deg = (((uint64_t)g.getData(src)) << 32) | cl;
	for (uint32_t i = pos + 1; i <= len; i ++) {//生成的边必须比之前生成的边大
		uint64_t deg =  (((uint64_t)g.getData(vid[einfo[i]])) << 32) | g.getData(vid[i]);
		uint64_t edge = (((uint64_t)vid[einfo[i]]) << 32) | vid[i];
		int deg_cmp = compare_edge(added_deg, deg);
		int cmp = compare_edge(added_edge, edge);
		if (deg_cmp < 0 || (deg_cmp == 0 && cmp <= 0))
			return false;
	}
	return true;
}//TODO the correctness of this need varification

__global__ void expand_kernel_by_edge(EmbeddingList emb_list, int level, CSRGraph g, emb_off_type base_off,
									  uint32_t f_size, KeyT *emb_vid, emb_off_type *emb_idx, label_type *emb_einfo,
									  uint32_t *counter, uint32_t *freq_edge_patterns) {
	uint32_t total_warp = (blockDim.x * gridDim.x)>>5;
	__shared__ KeyT sh_emb[BLOCK_SIZE/32][embedding_max_length];//sh_mem cache for warp-level embedding
	__shared__ label_type sh_edge_infos[BLOCK_SIZE/32][embedding_max_length];//sh_mem cache for warp-level edge infos
	uint32_t warp_id = (threadIdx.x + blockDim.x*blockIdx.x)/32;
	uint32_t lane_id = threadIdx.x%32;
	uint32_t write_chunk_off = 0, inside_chunk_off = 0;
	//warps get their inital space for frontier expand
	if (lane_id == 0) {
		write_chunk_off = atomicAdd(counter, warp_write_chunk);
		assert(write_chunk_off < EMB_FTR_CACHE_SIZE);
	}
	__syncwarp();
	write_chunk_off = __shfl_sync(0xffffffff, write_chunk_off, 0);
	//allocate embeddeing intermediate results for warps
	//TODO: here we assume f_size is multiple of warp_process_size, and this can be implmented later in our codes.
	for (uint32_t _i = warp_id; _i < f_size; _i += total_warp) {
		KeyT * local_emb = sh_emb[threadIdx.x/32];
		label_type *local_einfos = sh_edge_infos[threadIdx.x/32];
		__syncwarp();
		if (lane_id == 0) {
			emb_list.get_edge_embedding(level, base_off+_i, local_emb, local_einfos);
		}
		__syncwarp();
		//TODO here we simply traversal all neighbors of all vertices in the embedding, but in fact, there must exist some branch-prune methods
		//TODO here we use the naive method for all vertices' neighbor traversal
		bool  valid_emb = true;
		for (int v = 0; v <= level; v ++) {
			if (local_emb[v] == 0xffffffff) {
				valid_emb = false;
				break;
			}
		}
		if (!valid_emb) 
			continue;
		for (int v = 0; v <= level; v ++) {
			KeyT cur_vertex = local_emb[v];
			uint32_t src_label = g.getData(cur_vertex);
			int nbr_size = g.getDegree(cur_vertex);
			//if (nbr_size > 1000)
			//	continue;
			KeyT *cur_nbr = g.getAdjListofSrc(cur_vertex, g.edge_begin(cur_vertex));
			for (int n = lane_id; n < (31+nbr_size)/32*32; n += 32) {
				uint32_t results = 0;
				uint32_t matched = 0;
				KeyT candidate = 0xffffffff;
				uint32_t dst_label = 0;
				if (n < nbr_size) {
					candidate = cur_nbr[n];
					dst_label = g.getData(candidate);
					int srcXdst = (src_label*max_label)+dst_label;
					if ((freq_edge_patterns[srcXdst/32] >> (srcXdst%32))&1 == 1 )
						matched = (emb_check(local_emb, local_einfos, v, level, cur_vertex, candidate, g) == true) ? 1 : 0;
					//matched = (candidate > local_emb[0]) ? 1 : 0;
				}
				__syncwarp();
				warp_reduce(lane_id, matched, results);//TODO whether this is too time-consuming?
				uint32_t total_valid_num = matched + results;
				total_valid_num = __shfl_sync(0xffffffff, total_valid_num, 31);
				//if (level == 0 && lane_id == 0 && total_valid_num != 32 && e + 32 < f_size)
				//	printf("the valid num is %d\n", total_valid_num);
				if (total_valid_num + inside_chunk_off >= warp_write_chunk) {
					for (uint32_t p = inside_chunk_off + lane_id; p < warp_write_chunk; p += 32) {
						emb_idx[write_chunk_off + p] = _i + base_off;
						emb_vid[write_chunk_off + p] = 0xffffffff;
						emb_einfo[write_chunk_off + p] = 0xff;
					}
					if (lane_id == 0) {
						write_chunk_off = atomicAdd(counter, warp_write_chunk);
						assert(write_chunk_off < EMB_FTR_CACHE_SIZE);
					}
					write_chunk_off = __shfl_sync(0xffffffff, write_chunk_off, 0);
					inside_chunk_off = 0;
				}
				if(matched == 1) {
					emb_idx[write_chunk_off + inside_chunk_off + results] = _i + base_off;
					emb_vid[write_chunk_off + inside_chunk_off + results] = candidate;
					emb_einfo[write_chunk_off + inside_chunk_off + results] = v;
				}
				inside_chunk_off += total_valid_num;
			}
		}	
	}
	return ;
}

void expand_dynamic_by_edge(CSRGraph &g, EmbeddingList &emb_list, int cur_level, uint32_t *freq_edge_patterns) {
	//TODO in this function, we are exploring reasonable BATCH_EXPAND_OFF and EMB_FTR_CACHE_SIZE; if this works in this function, it should be applied to the whole framework later.
	Clock exp("expand edge embeddings in dynamic");
	exp.start();
	emb_off_type last_level_size = emb_list.size(cur_level-1);
	log_info("the last level size is %lu", last_level_size);
	uint32_t _expand_batch_size = EMB_FTR_CACHE_SIZE/6000;//TODO this is to be modified
	uint32_t batch_num = (last_level_size + _expand_batch_size-1)/_expand_batch_size;
	log_info("the batch num is %d", batch_num);
	emb_off_type *batch_expand_off = new emb_off_type [batch_num+1];
	memset(batch_expand_off, 0, sizeof(emb_off_type)*(batch_num+1));
	//KeyT **emb_vid_h = new KeyT *[batch_num];//host vid cache
	//emb_off_type **emb_idx_h = new emb_off_type *[batch_num];//host idx cache
	//label_type **emb_einfo_h = new label_type *[batch_num];//host einfo cache
	uint32_t nblocks = 3000;
	KeyT *emb_vid_d;//maximum space preallocated in GPU, and all warps ask for space chucks in dynamic
	emb_off_type *emb_idx_d;//maximum space preallocated in GPU, and all warps ask for space chucks in dynamic
	label_type *emb_einfo_d;//maximun space preallocated in GPU, and all warps ask for space chucks in dynamic
	check_cuda_error(cudaMalloc((void **)&emb_vid_d, sizeof(KeyT)*EMB_FTR_CACHE_SIZE));
	check_cuda_error(cudaMalloc((void **)&emb_idx_d, sizeof(emb_off_type)*EMB_FTR_CACHE_SIZE));
	check_cuda_error(cudaMalloc((void **)&emb_einfo_d, sizeof(label_type)*EMB_FTR_CACHE_SIZE));
	emb_list.add_level(0, cur_level);
	uint32_t *global_counter;
	check_cuda_error(cudaMalloc((void **)&global_counter, sizeof(uint32_t)));
	log_info("start expand kernel");
	emb_off_type write_base = 0;
	emb_off_type valid_unit_num = 0;
	for (uint32_t i = 0; i < batch_num; i ++) {
		check_cuda_error(cudaMemset(global_counter, 0, sizeof(uint32_t)));
		check_cuda_error(cudaMemset(emb_vid_d, -1, sizeof(KeyT)*EMB_FTR_CACHE_SIZE));
		emb_off_type base_off = (emb_off_type)_expand_batch_size*i;
		uint32_t cur_batch_size = (i < batch_num-1)? _expand_batch_size:(last_level_size-base_off);
		//if (i%10 == 0) log_info("start processing chunk %d",i);
		expand_kernel_by_edge<<<nblocks, BLOCK_SIZE>>>(emb_list, cur_level-1, g, base_off, cur_batch_size,
											   emb_vid_d, emb_idx_d, emb_einfo_d, global_counter, freq_edge_patterns);
		check_cuda_error(cudaDeviceSynchronize());
		//log_info("end kernel for chuck %d",i);
		check_cuda_error(cudaMemcpy(batch_expand_off+i, global_counter, sizeof(uint32_t),cudaMemcpyDeviceToHost));
		//assert(batch_expand_off[i] < EMB_FTR_CACHE_SIZE);
		//emb_vid_h[i] = (KeyT *)malloc(sizeof(KeyT)*batch_expand_off[i]);
		//emb_idx_h[i] = (emb_off_type *)malloc(sizeof(emb_off_type)*batch_expand_off[i]);
		//emb_einfo_h[i] = (label_type *)malloc(sizeof(label_type)*batch_expand_off[i]);
		//TODO streaming this 
		valid_unit_num += emb_list.check_valid_num(emb_vid_d, batch_expand_off[i]);
		emb_list.copy_to_vid_from_d(emb_vid_d, write_base, batch_expand_off[i], cur_level);
		emb_list.copy_to_idx_from_d(emb_idx_d, write_base, batch_expand_off[i], cur_level);
		emb_list.copy_to_einfo_from_d(emb_einfo_d, write_base, batch_expand_off[i], cur_level);
		write_base += batch_expand_off[i];
		//check_cuda_error(cudaMemcpy(emb_vid_h[i], emb_vid_d, sizeof(KeyT)*batch_expand_off[i], cudaMemcpyDeviceToHost));
		//check_cuda_error(cudaMemcpy(emb_idx_h[i], emb_idx_d, sizeof(emb_off_type)*batch_expand_off[i], cudaMemcpyDeviceToHost));
		//check_cuda_error(cudaMemcpy(emb_einfo_h[i], emb_einfo_d, sizeof(label_type)*batch_expand_off[i], cudaMemcpyDeviceToHost));
		check_cuda_error(cudaDeviceSynchronize());
	}
	log_info("end expand kernel");
	emb_list.size_adjustment();
	if (1) {
		uint64_t all_size = 0;
		all_size += g.nnodes * 8;
		all_size += g.nedges * 4 * 3;
		uint64_t embedTableSize = 0;
		for (int i = 0; i < emb_list.level(); i ++)
			embedTableSize += emb_list.size(i);
		uint64_t expand_results = thrust::reduce(thrust::host, batch_expand_off, batch_expand_off + batch_num);
		all_size += expand_results*(8+4+1+1 + 8+1 + 4);
		embedTableSize *= (8+4+1+1);
		if (embedTableSize*0.8 > 5000000000) {
			embedTableSize += 5000000000;
		} else {
			embedTableSize *= 1.8;
		}
		all_size += embedTableSize;
		uint32_t *gc_h = new uint32_t [1];
		cudaMemcpy(gc_h, global_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		embedTableSize += gc_h[0] * 13;
		delete [] gc_h;
		printf("all used mem is %d MB, and data label is %d MB\n", all_size/1024/1024, g.nnodes/1024/1024);
	}
	/*printf("here are edges recovered by expand:\n");
	for (int i = 0; i < batch_expand_off[0]; i++) {
		uint32_t sec = emb_vid_h[0][i], first = emb_idx_h[0][i];
		if (sec != 0xffffffff && first != 0xffffffff)
			printf("edges : %d %d\n", first, sec);
	}*/
	check_cuda_error(cudaFree(emb_vid_d));
	check_cuda_error(cudaFree(emb_idx_d));
	check_cuda_error(cudaFree(emb_einfo_d));
	check_cuda_error(cudaFree(global_counter));
	log_info(exp.count("end expand here, and valid emb number is %lu", valid_unit_num));
	/*thrust::exclusive_scan(thrust::host, batch_expand_off, batch_expand_off+batch_num+1, batch_expand_off);
	log_info(exp.count("end expand embeddings by edge, and new ftr size is %lu", batch_expand_off[batch_num]));
	emb_list.add_level(batch_expand_off[batch_num], cur_level);
	for (uint32_t i = 0; i < batch_num; i ++) {
		emb_off_type base_off = batch_expand_off[i];
		emb_off_type ftr_size = batch_expand_off[i+1] - batch_expand_off[i];
		//emb_list.copy_to_vid_from_h(emb_vid_h[i], base_off, ftr_size, cur_level);
		//emb_list.copy_to_idx_from_h(emb_idx_h[i], base_off, ftr_size, cur_level);
		//emb_list.copy_to_einfo_from_h(emb_einfo_h[i], base_off, ftr_size, cur_level);
		free(emb_vid_h[i]);
		free(emb_idx_h[i]);
		free(emb_einfo_h[i]);
	}
	log_info(exp.count("end copy results"));
	delete [] emb_vid_h;
	delete [] emb_idx_h;
	delete [] emb_einfo_h;*/
	delete [] batch_expand_off;
	return ;
}//in this expand function, the whole frontier are expanded in batches; what's more, we give local shared memory to each warp to cache local intersection results; lastly, we use dynamically methods to place the newly generated frontiers.


#endif
