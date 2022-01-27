#ifndef AGGREGRATE_H
#define AGGREGRATE_H
#include "graph.cuh"
#include "embedding.cuh"
__device__ patternID emb2pattern_id(KeyT *emb, label_type *einfo, uint32_t *vlabel, int len, CSRGraph g) {
	//Here is how we use vlabel to keep all charactors we use
	//for dulicated vids, their 32-bit wide data is used as : 0xf(real data pos)fffffff(all set to 1)
	//for real data, 32-bit data is used as: 0xf(final vertex order) ffff (vertex encoding) f (degree) ff(label)
	//"emb" is only used for vertex duplicate check, after that, it is used for vertex encoding
	//memset(vlabel, 0, sizeof(uint32_t)*len);
	//TODO bit usage has not been changed after we enlarge the bit number of patternID
	vlabel[0] = 0;
	//count the number of distinct vertex
	int distinct_v = 1;
	//int e1 = emb[0], e2 = emb[1];
	for (int i = 1; i < len; i ++) {
		vlabel[i] = 0;
		for (int j = 0; j < i; j ++) {//find the first same vid
			if (emb[i] == emb[j]) {
				vlabel[i] |= 0xfffffff;
				vlabel[i] |= (j<<28);
				break;
			} 
		}
		if ((vlabel[i] &0xfffffff) != 0xfffffff)
			distinct_v ++;
	}
	//NOTE: the valid number of all vertex is not consecutive
	//encoding vertices
		//label collection and degree counting (label_num < 256)
	for (int i = 0; i < len; i ++) {
		if ((vlabel[i]&0xfffffff) == 0xfffffff)
			continue;
		vlabel[i] = g.getData(emb[i]);
	}
	for (int i = 1; i < len; i ++) {
		int real_dst_pos = (vlabel[i]&0xfffffff) == 0xfffffff ? (vlabel[i] >> 28) : i;
		int real_src_pos = (vlabel[einfo[i]]&0xfffffff) == 0xfffffff ? (vlabel[einfo[i]] >> 28): einfo[i];
		vlabel[real_src_pos] += (1<<8);
		vlabel[real_dst_pos] += (1<<8);
	}
		//set initial encoding
	for (int i = 0; i < len; i ++) {
		if ((vlabel[i] &0xfffffff) == 0xfffffff)
			continue;
		uint32_t value =((vlabel[i]&0xff)+1)*(((vlabel[i]>>8)&0xf)+1);
		vlabel[i] += (value<<12);
	}
		//iterativly encoding
	uint32_t buffer[embedding_max_length];
	uint32_t max_it = len-1 > 3 ? 3 : len - 1;
	for (int it = 0; it < max_it; it ++) {
		memcpy(buffer, vlabel, sizeof(uint32_t)*len);
		for (int i = 1; i < len; i ++) {
			int real_dst_pos = (vlabel[i]&0xfffffff) == 0xfffffff ? (vlabel[i] >> 28) : i;
			int real_src_pos = (vlabel[einfo[i]]&0xfffffff) == 0xfffffff ? (vlabel[einfo[i]] >> 28): einfo[i];

			int encoding = (buffer[real_dst_pos] >> 12) &0xffff;
			int deg = (buffer[real_dst_pos] >> 8)&0xf;
			vlabel[real_src_pos] += ((encoding/deg)<<12)/(it+2);
			encoding = (buffer[real_src_pos] >> 12) &0xffff;
			deg = (buffer[real_src_pos] >> 8)&0xf;
			vlabel[real_dst_pos] += ((encoding/deg)<<12)/(it+2);
		}
	}
	//calculate vertex order by vertex encoding
	int vertex_order = 0xf;
	for (int i = 0; i < len; i ++) {
		if ((vlabel[i]&0xfffffff) == 0xfffffff)
			continue;
		vlabel[i] += (vertex_order << 28);
	}
	for (uint32_t i = 0; i < distinct_v; i ++) {
		uint32_t max_encoding = 0;
		uint32_t max_encoding_pos = 0;
		for (int j = 0; j < len; j ++) {
			if ((vlabel[j]&0xfffffff) == 0xfffffff)
				continue;
			if (((vlabel[j]>>28)&0xf) != 0xf)
				continue;
			uint32_t my_encoding = (vlabel[j]&0xfffffff);
			if (my_encoding > max_encoding) {
				max_encoding_pos = j;
				max_encoding = my_encoding;
			}
		}
		vlabel[max_encoding_pos] = (i << 28) | (vlabel[max_encoding_pos] &0xfffffff);
	}
	/*int f1 = (vlabel[0]>>28)&0xf > (vlabel[1]>>28)&0xf ? 1: -1;
	int f2 = (g.getData(emb[0]) < g.getData(emb[1])) ? 1 : -1;
	if (f1*f2 == -1) {
		printf("%x %x %x %x %x %x\n",(vlabel[0]>>28)&0xf, (vlabel[1]>>28)&0xf, g.getData(emb[0]), g.getData(emb[1]), vlabel[0]&0xfffffff, vlabel[1]&0xfffffff) ;
	}*/
	//generate the pattern_id
	//64bit pattern id : 7+6+5+4+3+2+1 + 4X8, so the maximum embedding is 7, and max label num is 15 by this
	patternID pattern_id(0, 0);
	//pattern_id.lab = (uint64_t)(g.getData(emb[0]) + (g.getData(emb[1])<<8));
	//return pattern_id;
	for (int i = 0; i < len; i ++) {
		int real_dst_pos = (vlabel[i]&0xfffffff) == 0xfffffff ? ((vlabel[i] >> 28)&0xf): i;
		//set label
		if (real_dst_pos == i) {
			int label = g.getData(emb[i]), _off = ((vlabel[i] >>28)&0xf)*8;
			pattern_id.lab += (label << _off);
		}
		if (i == 0)
			continue;
		int real_src_pos = (vlabel[einfo[i]]&0xfffffff) == 0xfffffff ? vlabel[einfo[i]] >> 28: einfo[i];
		int src_v_order = (vlabel[real_src_pos] >> 28)& 0xf;
		int dst_v_order = (vlabel[real_dst_pos] >> 28)& 0xf;
		if (src_v_order > dst_v_order) {
			int t = src_v_order; src_v_order = dst_v_order; dst_v_order = t;
		}
		int off = (7+8-src_v_order)*src_v_order/2 + (dst_v_order-src_v_order-1);
		pattern_id.nbr = pattern_id.nbr | (1 << off);
		//if (g.getData(emb[0]) == 0 && g.getData(emb[1]) == 1) 
		//	printf("the labels is %lx\n", pattern_id.lab);
		//if (len == 2 && g.getData(emb[0]) == 0)
		//	printf("%d %d label : 0 %d\n", emb[0], emb[1], g.getData(emb[1]));
	}
	return pattern_id;
}
__global__ void map_emb2pid(CSRGraph g, EmbeddingList emblist, patternID *pids, 
				            int level, emb_off_type base_off, uint32_t emb_size) {
		__shared__ KeyT local_emb[BLOCK_SIZE][embedding_max_length];
		__shared__ label_type local_einfo[BLOCK_SIZE][embedding_max_length];
		__shared__ uint32_t vlabel[BLOCK_SIZE][embedding_max_length];
		int idx = threadIdx.x + blockDim.x*blockIdx.x;
		int tid = threadIdx.x;
		for (int i = idx; i < emb_size; i += (blockDim.x*gridDim.x)) {
			emblist.get_edge_embedding(level, base_off+i, local_emb[tid], local_einfo[tid]);
			//if (i < 30) 
			//	printf("%d th emb, %d %d %d %d\n", i, local_emb[tid][0], local_emb[tid][1], g.getData(local_emb[tid][0]), g.getData(local_emb[tid][1]));
			bool valid_emb = true;
			for (uint32_t j = 0; j <= level; j ++) {
				if(local_emb[tid][j] == 0xffffffff) {
					valid_emb = false;
					break;
				}
			}
			patternID pattern_id(-1, (uint64_t)-1);
			if (valid_emb) {
				pattern_id = emb2pattern_id(local_emb[tid], local_einfo[tid], vlabel[tid], level+1, g);
			}
			pids[base_off+i] = pattern_id;
		}
		return ;
}

void map_embeddings_to_pids(CSRGraph g, EmbeddingList emb_list, patternID *pattern_ids, int level) {
		uint64_t emb_nums = emb_list.size(level);
		uint32_t batch_num = (emb_nums+expand_batch_size-1)/expand_batch_size;
	check_cuda_error(cudaDeviceSynchronize());
	//printf("the batch num of map embedding to pids is %d\n", batch_num);
	for (uint32_t i = 0; i < batch_num; i++) {
		emb_off_type base_off = (uint64_t)i*expand_batch_size;
		uint32_t cur_size = emb_nums - base_off;
		cur_size = cur_size > expand_batch_size ? expand_batch_size : cur_size;
		uint32_t num_blocks = 10000;
		map_emb2pid<<<num_blocks, BLOCK_SIZE>>>(g, emb_list, pattern_ids, level, base_off, cur_size);
		check_cuda_error(cudaDeviceSynchronize());
		//if (i%10 == 0) log_info("the %dth batch of embedding is done", i+1);
	}
	log_info("map_embedding_to_pids is done");
	//emb_list.display(level, emb_nums);	
	//code validation
	/*patternID *h_pids = new patternID [emb_nums];
	check_cuda_error(cudaMemcpy(h_pids, pattern_ids, sizeof(patternID)*emb_nums, cudaMemcpyDeviceToHost));
	for (int p = 0; p < emb_nums; p ++)
		printf("%d %d\n", h_pids[p].lab&0xff, (h_pids[p].lab>>8)&0xff);
	delete [] h_pids;*/
	//emb_list.check_all(level, emb_nums, pattern_ids, g);
	//sample mapped pattern
	/*int sample_size = 40;
	uint64_t *patterns_h = new uint64_t [sample_size];
	for (int i = 0; i < sample_size; i ++) {
		check_cuda_error(cudaMemcpy(patterns_h+i, pattern_ids+emb_nums/sample_size*i, sizeof(uint64_t), cudaMemcpyDeviceToHost));
		printf("%lx\n", patterns_h[i]);
	}
	delete [] patterns_h;*/
	return ;
}


__global__ void count_frequent_pattern(patternID *pattern_ids, emb_off_type pid_size, int threshold, 
									   uint32_t *fre_pattern_num, uint8_t *stencil) {
	uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
	int local_fre_pattern = 0;
	for (emb_off_type i = tid; i <= pid_size-threshold; i += (blockDim.x*gridDim.x)) {
		if (i == 0 || !(pattern_ids[i] == pattern_ids[i-1])) {
			if (pattern_ids[i] == pattern_ids[i+threshold-1]) {
				stencil[i] = 1;
				local_fre_pattern ++;
			}
		}
	}
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 16);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 8);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 4);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 2);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 1);
	if (threadIdx.x%32 == 0) {
		fre_pattern_num[tid/32] = local_fre_pattern;
	}
	return ;
}

__global__ void emb_validation_check(EmbeddingList emb_list, emb_off_type emb_size, patternID *fre_patterns,
									 uint32_t fre_pattern_num, uint8_t *valid_embs, int level, 
									 emb_off_type base_off, uint32_t *counter, CSRGraph g) {
	__shared__ KeyT local_emb[BLOCK_SIZE][embedding_max_length];
	__shared__ label_type local_einfo[BLOCK_SIZE][embedding_max_length];
	__shared__ uint32_t vlabel[BLOCK_SIZE][embedding_max_length];
	int thread_id = threadIdx.x+blockIdx.x*blockDim.x;
	int idx = threadIdx.x;
	uint32_t local_count = 0;
	for (uint32_t _i = thread_id; _i < emb_size; _i += (blockDim.x*gridDim.x)) {
		emb_list.get_edge_embedding(level, base_off+_i, local_emb[idx], local_einfo[idx]);
		bool valid_emb = true;
		for (uint32_t j = 0; j <= level; j ++) {
			if(local_emb[idx][j] == 0xffffffff) {
				valid_emb = false;
				break;
			}
		}
		if (valid_emb) {
			patternID pattern_id = emb2pattern_id(local_emb[idx], local_einfo[idx], vlabel[idx], level+1, g);
			if (binarySearch<patternID>(fre_patterns, fre_pattern_num, pattern_id) != -1) {
				valid_embs[ _i + base_off] = 1;
				local_count ++;
			}
		}
	}
	local_count += __shfl_down_sync(0xffffffff, local_count, 16);
	local_count += __shfl_down_sync(0xffffffff, local_count, 8);
	local_count += __shfl_down_sync(0xffffffff, local_count, 4);
	local_count += __shfl_down_sync(0xffffffff, local_count, 2);
	local_count += __shfl_down_sync(0xffffffff, local_count, 1);
	if (threadIdx.x%32 == 0)
		counter[thread_id/32] = local_count;
	return ;

}
struct cmp_pid {
	__device__ __host__ bool operator() (const patternID& p1, const patternID& p2) {
		return p1.nbr < p2.nbr || (p1.nbr == p2.nbr && p1.lab < p2.lab);
	}
};
__global__ void set_freq_edge_pattern(EmbeddingList emb_list, emb_off_type emb_size, uint32_t l, uint32_t *freq_edge_patterns, CSRGraph g) {
	__shared__ KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
	int thread_id = threadIdx.x+ blockDim.x*blockIdx.x;
	KeyT *local_emb = sh_emb[threadIdx.x];
	for (emb_off_type i = thread_id; i < emb_size; i += (blockDim.x*gridDim.x)) {
		emb_list.get_embedding(l, i, local_emb);
		if (local_emb[0] == 0xffffffff || local_emb[1] == 0xffffffff)
			continue;
		uint32_t src_label = g.getData(local_emb[0]), dst_label = g.getData(local_emb[1]);
		int multiple = (src_label * max_label) + dst_label;
		atomicOr(freq_edge_patterns + multiple/32, 1<<(multiple%32));
		multiple = (dst_label * max_label) + src_label;
		atomicOr(freq_edge_patterns + multiple/32, 1<<(multiple%32));
	}
	return ;
}
void aggregrate_and_filter(CSRGraph g, EmbeddingList emb_list, patternID *pattern_ids, int level, int threshold, uint32_t *freq_edge_patterns) {
	//sort all pattern_ids
	//WARNING: this may cause all embedding list thrash bettween cpu and gpu, but that's affordable
	emb_off_type pid_size = emb_list.size(level);
	log_info("start sort embedding ids... ...");
	thrust::sort(thrust::device, pattern_ids, pattern_ids + pid_size, cmp_pid());//TODO: out of memory?
	log_info("sort all embedding ids done");	
	//filter out all pattern ids whose support satisfy the threshold
	uint32_t block_num = 10000;
	uint32_t *fre_pattern_num;
	uint8_t *stencil;
	check_cuda_error(cudaMalloc((void **)&stencil, pid_size*sizeof(uint8_t)));
	check_cuda_error(cudaMemset(stencil, 0, sizeof(uint8_t)*pid_size));
	check_cuda_error(cudaMalloc((void **)&fre_pattern_num, BLOCK_SIZE/32*sizeof(uint32_t)*block_num));
	check_cuda_error(cudaMemset(fre_pattern_num, 0, BLOCK_SIZE/32*sizeof(uint32_t)*block_num));
	//TODO here we assume all pattern_ids and stential can be put on the device, and no batch process
	count_frequent_pattern<<<block_num, BLOCK_SIZE>>>(pattern_ids, pid_size, threshold, fre_pattern_num, stencil);
	check_cuda_error(cudaDeviceSynchronize());
	uint32_t total_fre_pattern = thrust::reduce(thrust::device, fre_pattern_num, fre_pattern_num + BLOCK_SIZE/32*block_num);//the number of valid patterns
	log_info("count frequent patterns done, total frequent pattern num is %d", total_fre_pattern);
	check_cuda_error(cudaFree(fre_pattern_num));
	patternID *fre_patterns;//frequent patterns
	check_cuda_error(cudaMalloc((void **)&fre_patterns, sizeof(patternID)*total_fre_pattern));
	thrust::copy_if(thrust::device, pattern_ids, pattern_ids + pid_size, stencil, fre_patterns, is_valid());
	check_cuda_error(cudaFree(stencil));
	check_cuda_error(cudaFree(pattern_ids));
	log_info("frequent pattern collection done");
	//filter out all embeddings whose pattern ids satisfy the threshold
	uint8_t *valid_emb;
	check_cuda_error(cudaMalloc((void **)&valid_emb, sizeof(uint8_t)*pid_size));
	check_cuda_error(cudaMemset(valid_emb, 0, sizeof(uint8_t)*pid_size));
	uint32_t batch_num = (pid_size + expand_batch_size -1)/expand_batch_size;
	uint32_t valid_emb_num = 0;
	uint32_t *d_counter;
	check_cuda_error(cudaMalloc((void **)&d_counter, BLOCK_SIZE/32*block_num*sizeof(uint32_t)));
	for (int i = 0; i < batch_num; i ++) {
		check_cuda_error(cudaMemset(d_counter, 0, BLOCK_SIZE/32*block_num*sizeof(uint32_t)));
		emb_off_type base_off = (emb_off_type)i*expand_batch_size;
		uint32_t cur_size = pid_size - base_off;
		cur_size = cur_size > expand_batch_size ? expand_batch_size : cur_size;
		emb_validation_check<<<block_num, BLOCK_SIZE>>>(emb_list, cur_size, fre_patterns, total_fre_pattern, valid_emb, level, base_off, d_counter, g);
		check_cuda_error(cudaDeviceSynchronize());
		valid_emb_num += thrust::reduce(thrust::device, d_counter, d_counter+BLOCK_SIZE/32*block_num);
	}
	log_info("embedding validation check done, and valid emb num for now is %d",valid_emb_num);
	check_cuda_error(cudaFree(d_counter));
	//embedding list compaction
	emb_list.compaction(level, valid_emb, valid_emb_num);
	check_cuda_error(cudaFree(valid_emb));	
	if (level == 1) {//here we set frequent edge pattern flags
		check_cuda_error(cudaMemset(freq_edge_patterns, 0, sizeof(uint32_t)*max_label*max_label/32));
		set_freq_edge_pattern<<<block_num, BLOCK_SIZE>>>(emb_list, emb_list.size(), level, freq_edge_patterns, g);
		check_cuda_error(cudaDeviceSynchronize());
	}
	log_info("embedding list compaction done");
}

#endif
