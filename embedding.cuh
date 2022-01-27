#ifndef EMBEDDING_CUH_
#define EMBEDDING_CUH_
//embedding only describe vertices (labels)
#include "graph.cuh"
struct SimpleEmbedding{

};
//embedding describing vertices (labels) and adjacency relationships
struct Embedding{

};
KeyT *all_vid;
emb_off_type *all_idx;
label_type *all_einfo;
emb_off_type mem_head;
__global__ void check_valid_ftr_num(KeyT *vids, emb_off_type *results, emb_off_type f_size) {
	uint32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t total_threads = gridDim.x * blockDim.x;
	emb_off_type local_results = 0;
	for (emb_off_type i = thread_idx; i < f_size; i += total_threads) {
		if (vids[i] != 0xffffffff)
			local_results ++;
	}
	__syncwarp();
	local_results += __shfl_down_sync(0xffffffff, local_results, 16);
	local_results += __shfl_down_sync(0xffffffff, local_results, 8);
	local_results += __shfl_down_sync(0xffffffff, local_results, 4);
	local_results += __shfl_down_sync(0xffffffff, local_results, 2);
	local_results += __shfl_down_sync(0xffffffff, local_results, 1);
	if (threadIdx.x %32 == 0)
		results[thread_idx/32] = local_results;
	return;
}

//TODO : currently, our embedding list are all putting in the unified memory or GPU memory
class EmbeddingList {
public:
	EmbeddingList() {}
	~EmbeddingList() {}
	void init(OffsetT initial_size, unsigned max_size , mem_type mem_tp, bool use_edge_label, bool use_dag = true) {
		last_level = 0;
		assert(max_size > 1);
		assert(mem_tp < 2);
		memory_type = mem_tp;
		max_level = max_size;
		enable_elabel = use_edge_label;
		h_vid_lists = (KeyT **)malloc(max_level * sizeof(KeyT*));
		h_idx_lists = (emb_off_type **)malloc(max_level * sizeof(emb_off_type*));
		for (int i = 0; i < max_size; i ++) {
			h_vid_lists[i] = NULL;
			h_idx_lists[i] = NULL;
		}
		check_cuda_error(cudaMalloc(&d_vid_lists, max_level * sizeof(KeyT*)));
		check_cuda_error(cudaMalloc(&d_idx_lists, max_level * sizeof(emb_off_type*)));
		if (enable_elabel) {
			h_edge_infos = (label_type**)malloc(max_level * sizeof(label_type*));
			check_cuda_error(cudaMalloc(&d_edge_infos, max_level * sizeof(label_type*)));
			for (int i = 0; i < max_size; i ++)
				h_edge_infos[i] = NULL;
		}
		sizes = new uint64_t [max_level];
		initial_size = (initial_size+31)/32*32;
		sizes[0] = initial_size;
		OffsetT nnz = initial_size;
		//if (!use_dag) nnz = nnz / 2;
		//allocate memory for the first layer
		switch (memory_type) {
			case GPU_MEM:
				check_cuda_error(cudaMalloc((void **)&h_vid_lists[0], nnz * sizeof(KeyT)));
				check_cuda_error(cudaMemset(h_vid_lists[0]+nnz-32, -1, sizeof(KeyT)*32));
				//check_cuda_error(cudaMalloc((void **)&h_idx_lists[0], nnz * sizeof(KeyT)));
				if (enable_elabel) {
					check_cuda_error(cudaMalloc((void **)&h_edge_infos[0], nnz * sizeof(label_type)));
					check_cuda_error(cudaMemset(h_edge_infos[0], -1, sizeof(label_type)*nnz));
				}
				break;
			case UNIFIED_MEM:
				check_cuda_error(cudaMallocManaged((void **)&all_vid, sizeof(KeyT)*MAX_EMB_UNIT_NUM));
				check_cuda_error(cudaMallocManaged((void **)&all_idx, sizeof(emb_off_type)*MAX_EMB_UNIT_NUM));
				check_cuda_error(cudaMemAdvise(all_vid, sizeof(KeyT)*MAX_EMB_UNIT_NUM, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
				check_cuda_error(cudaMemAdvise(all_idx, sizeof(emb_off_type)*MAX_EMB_UNIT_NUM, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
				h_vid_lists[0] = all_vid;
				memset(h_vid_lists[0]+nnz-32, -1, sizeof(KeyT)*32);
            	//check_cuda_error(cudaMemAdvise(h_vid_lists[0], nnz*sizeof(KeyT), cudaMemAdviseSetReadMostly, 0));
				//check_cuda_error(cudaMallocManaged((void **)&h_idx_lists[0], nnz * sizeof(KeyT)));
				if (enable_elabel) {
					check_cuda_error(cudaMallocManaged((void **)&all_einfo, sizeof(label_type)*MAX_EMB_UNIT_NUM));
					check_cuda_error(cudaMemAdvise(all_einfo, sizeof(label_type)*MAX_EMB_UNIT_NUM, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
					h_edge_infos[0] = all_einfo;
					memset(h_edge_infos[0], -1, sizeof(label_type)*nnz);
				}
				mem_head += nnz;
				break;
		}
		//only the first element in h_vid_lists is assigned, but all copy to d_vid_list
		check_cuda_error(cudaMemcpy(d_vid_lists, h_vid_lists, max_level * sizeof(KeyT*), cudaMemcpyHostToDevice));
		//check_cuda_error(cudaMemcpy(d_idx_lists, h_idx_lists, max_level * sizeof(KeyT*), cudaMemcpyHostToDevice));
		if (enable_elabel)
			check_cuda_error(cudaMemcpy(d_edge_infos, h_edge_infos, max_level * sizeof(label_type*), cudaMemcpyHostToDevice));
		
	}
	void display(int level, int display_num) {
		KeyT **ho_vid_lists = new KeyT *[level+1];
		emb_off_type **ho_idx_lists = new emb_off_type *[level+1];
		ho_vid_lists[0] = new KeyT [display_num];
		if (level >= 1) {
			ho_vid_lists[1] = new KeyT [display_num];
			ho_idx_lists[1] = new emb_off_type [display_num];
			memcpy(ho_vid_lists[1], h_vid_lists[1], sizeof(KeyT)*display_num);
			memcpy(ho_idx_lists[1], h_idx_lists[1], sizeof(emb_off_type)*display_num);
		}
		memcpy(ho_vid_lists[0], h_vid_lists[0], sizeof(KeyT)*display_num);
		printf("show %u embeddings:\n", display_num);
		for (int i = 0; i < display_num; i ++) {
			if (level >= 1)
				printf("%u %u\n", ho_vid_lists[0][ho_idx_lists[1][i]], ho_vid_lists[1][i]);
			else 
				printf("%u\n", ho_vid_lists[0][i]);
		}

		delete [] ho_vid_lists[0];
		if (level >= 1) {
			delete [] ho_idx_lists[1];
			delete [] ho_vid_lists[1];
		}
		delete [] ho_idx_lists;
		delete [] ho_vid_lists;
		return;
	}
	void display_valid(int level, int display_num) {
		KeyT **ho_vid_lists = new KeyT *[level+1];
		emb_off_type **ho_idx_lists = new emb_off_type *[level+1];
		ho_vid_lists[0] = new KeyT [display_num];
		if (level >= 1) {
			ho_vid_lists[1] = new KeyT [display_num];
			ho_idx_lists[1] = new emb_off_type [display_num];
			memcpy(ho_vid_lists[1], h_vid_lists[1], sizeof(KeyT)*display_num);
			memcpy(ho_idx_lists[1], h_idx_lists[1], sizeof(emb_off_type)*display_num);
		}
		memcpy(ho_vid_lists[0], h_vid_lists[0], sizeof(KeyT)*display_num);
		printf("show %u embeddings:\n", display_num);
		for (int i = 0; i < display_num; i ++) {
			if (level >= 1) { 
				KeyT first = ho_vid_lists[0][ho_idx_lists[1][i]], sec = ho_vid_lists[1][i];
				if (first == 0xffffffff || sec == 0xffffffff)
					continue;
				printf("%d %d\n", first, sec);
			}
			else {
				if (ho_vid_lists[0][i] != 0xffffffff) 
					printf("%d\n", ho_vid_lists[0][i]);
			}
		}

		delete [] ho_vid_lists[0];
		if (level >= 1) {
			delete [] ho_idx_lists[1];
			delete [] ho_vid_lists[1];
		}
		delete [] ho_idx_lists;
		delete [] ho_vid_lists;
		return;
	}
	void check_all(int level, uint64_t display_num, patternID *pattern_ids, CSRGraph g) {
		KeyT **ho_vid_lists = new KeyT *[level+1];
		emb_off_type **ho_idx_lists = new emb_off_type *[level+1];
		ho_vid_lists[0] = new KeyT [display_num];
		if (level >= 1) {
			ho_vid_lists[1] = new KeyT [display_num];
			ho_idx_lists[1] = new emb_off_type [display_num];
			memcpy(ho_vid_lists[1], h_vid_lists[1], sizeof(KeyT)*display_num);
			memcpy(ho_idx_lists[1], h_idx_lists[1], sizeof(emb_off_type)*display_num);
		}
		memcpy(ho_vid_lists[0], h_vid_lists[0], sizeof(KeyT)*display_num);
		patternID *h_pid = new patternID [display_num];
		cudaMemcpy(h_pid, pattern_ids, sizeof(patternID)*display_num, cudaMemcpyDeviceToHost);
		uint32_t nodeNum = g.nnodes;
		node_data_type *nodeData = new node_data_type [nodeNum];
		cudaMemcpy(nodeData, g.node_data, sizeof(node_data_type)*nodeNum, cudaMemcpyDeviceToHost);
		printf("show %u embeddings:\n", display_num);
		for (int i = 0; i < display_num; i ++) {
			if (level >= 1) { 
				KeyT first = ho_vid_lists[0][ho_idx_lists[1][i]], sec = ho_vid_lists[1][i];
				if (first == 0xffffffff || sec == 0xffffffff)
					continue;
				uint64_t lab = h_pid[i].lab;
				printf("node %d %d data %d %d id_data %d %d\n", first, sec, nodeData[first], nodeData[sec], lab&0xff, (lab>>8)&0xff);
			}
			else {
				if (ho_vid_lists[0][i] != 0xffffffff) 
					printf("%d\n", ho_vid_lists[0][i]);
			}
		}

		delete [] ho_vid_lists[0];
		if (level >= 1) {
			delete [] ho_idx_lists[1];
			delete [] ho_vid_lists[1];
		}
		delete [] ho_idx_lists;
		delete [] ho_vid_lists;
		delete [] nodeData;
		delete [] h_pid;
		return;
	}
	__device__ __host__ inline KeyT get_vid(unsigned level, emb_off_type id) const { return d_vid_lists[level][id]; }
	__device__ __host__ inline KeyT get_idx(unsigned level, emb_off_type id) const { return d_idx_lists[level][id]; }
	__device__ __host__ label_type get_edge_info(unsigned level, KeyT id) const { return d_edge_infos[level][id]; }
	__device__ inline void set_edge_info(unsigned level, emb_off_type id, label_type l) {d_edge_infos[level][id] = l; }
	//__device__ unsigned get_pid(KeyT id) const { return adjacency list[id]; }//TODO what's pid????
	__device__ inline void set_vid(unsigned level, emb_off_type id, KeyT vid) { d_vid_lists[level][id] = vid; }
	__device__ inline void set_idx(unsigned level, emb_off_type id, emb_off_type idx) { d_idx_lists[level][id] = idx; }
	//__device__ void set_pid(KeyT id, unsigned pid) { adjacency list[id] = pid; }
	size_t size() const { return sizes[last_level]; }
	uint32_t level() const {return last_level;}
	size_t size(unsigned level) const { return sizes[level]; }
	//__device__ VertexList get_vid_list(unsigned level) { return vid_lists[level]; }
	//__device__ UintList get_idx_list(unsigned level) { return idx_lists[level]; }
	//__device__ ByteList get_his_list(unsigned level) { return his_lists[level]; }
	void add_level(emb_off_type size, uint32_t level) { 
		//WARNING: this function works in a different way for GPU_MEM and UNIFIED_MEM:
		//GPU_MEM : allocation designed-sized memory
		//UNIFIED_MEM : not allocate, just pointer embedding pointer to the corresponding all_ptr
		last_level = level;
		size = (size+31)/32*32;
		//log_info("we have just alloc the level %d", last_level);
		assert(last_level < max_level);
		switch (memory_type) {
			case GPU_MEM:
				check_cuda_error(cudaMalloc((void **)&h_vid_lists[last_level], size * sizeof(KeyT)));
				check_cuda_error(cudaMalloc((void **)&h_idx_lists[last_level], size * sizeof(emb_off_type)));
				//TODO: here we set size of frontiers multiple of 32
				check_cuda_error(cudaMemset(h_vid_lists[last_level]+size-32, -1, sizeof(KeyT)*32));
				check_cuda_error(cudaMemset(h_idx_lists[last_level]+size-32, 0, sizeof(emb_off_type)*32));
				if (enable_elabel) {
					check_cuda_error(cudaMalloc((void **)&h_edge_infos[last_level], size * sizeof(label_type)));
					check_cuda_error(cudaMemset(h_edge_infos[last_level]+size-32, -1,sizeof(label_type)*32));
				}
				sizes[last_level] = size;
				break;
			case UNIFIED_MEM:
				h_vid_lists[last_level] = all_vid + mem_head;
				h_idx_lists[last_level] = all_idx + mem_head;
            	//check_cuda_error(cudaMemAdvise(h_vid_lists[last_level], size*sizeof(KeyT), cudaMemAdviseSetReadMostly, 0));
            	//check_cuda_error(cudaMemAdvise(h_idx_lists[last_level], size*sizeof(emb_off_type), cudaMemAdviseSetReadMostly, 0));
				if (enable_elabel) {
					h_edge_infos[last_level] = all_einfo + mem_head;
				}
				sizes[last_level] = 0;
				break;
		}
		check_cuda_error(cudaMemcpy(d_vid_lists, h_vid_lists, max_level * sizeof(KeyT*), cudaMemcpyHostToDevice));
		check_cuda_error(cudaMemcpy(d_idx_lists, h_idx_lists, max_level * sizeof(emb_off_type*), cudaMemcpyHostToDevice));
		if (enable_elabel)
			check_cuda_error(cudaMemcpy(d_edge_infos, h_edge_infos, max_level * sizeof(history_type*), cudaMemcpyHostToDevice));
	}
	void compaction(unsigned l, uint8_t *valid_emb, uint32_t valid_emb_size) {
		//TODO here we firstly assume that this function will only be used when VERTEX LABEL and EINFO are enabled
		KeyT *new_vid;
		emb_off_type* new_idx;
		label_type *new_einfo;
		emb_off_type old_emb_size = size(l);
		valid_emb_size = (valid_emb_size + 31)/32*32;
		switch (memory_type) {
				case GPU_MEM:
					printf("COMPACTION ON DEVICE MEMORY IS NOT SUPPORT YET\n");
					/*check_cuda_error(cudaMalloc((void **)&new_vid, sizeof(KeyT)*valid_emb_size));
					check_cuda_error(cudaMemset(new_vid + valid_emb_size - 32, -1, sizeof(KeyT)*32));
					check_cuda_error(cudaMalloc((void **)&new_idx, sizeof(emb_off_type)*valid_emb_size));
					check_cuda_error(cudaMemset(new_idx + valid_emb_size - 32, 0, sizeof(emb_off_type)*32));
					check_cuda_error(cudaMalloc((void **)&new_einfo, sizeof(label_type)*valid_emb_size));
					thrust::copy_if(thrust::device, h_vid_lists[l], h_vid_lists[l] + old_emb_size,
													valid_emb, new_vid, is_valid());
					thrust::copy_if(thrust::device, h_idx_lists[l], h_idx_lists[l] + old_emb_size,
													valid_emb, new_idx, is_valid());
					thrust::copy_if(thrust::device, h_edge_infos[l], h_edge_infos[l] + old_emb_size,
													valid_emb, new_einfo, is_valid());*/
					break;
				case UNIFIED_MEM:
					check_cuda_error(cudaMallocManaged((void **)&new_vid, sizeof(KeyT)*valid_emb_size));
					memset(new_vid + valid_emb_size -32, -1, sizeof(KeyT)*32);
					//TODO modify get_edge_embedding to obatin invalid embeddings
					check_cuda_error(cudaMallocManaged((void **)&new_idx, sizeof(emb_off_type)*valid_emb_size));
					memset(new_idx + valid_emb_size -32, 0, sizeof(emb_off_type)*32);
					thrust::copy_if(thrust::device, h_vid_lists[l], h_vid_lists[l] + old_emb_size,valid_emb, new_vid, is_valid());
					thrust::copy_if(thrust::device, h_idx_lists[l], h_idx_lists[l] + old_emb_size,valid_emb, new_idx, is_valid());
					check_cuda_error(cudaMemcpy(h_vid_lists[l], new_vid, sizeof(KeyT)*valid_emb_size, cudaMemcpyHostToHost));
					check_cuda_error(cudaMemcpy(h_idx_lists[l], new_idx, sizeof(emb_off_type)*valid_emb_size, cudaMemcpyHostToHost));
					
					if(enable_elabel) {
						check_cuda_error(cudaMallocManaged((void **)&new_einfo, sizeof(label_type)*valid_emb_size));
						thrust::copy_if(thrust::device, h_edge_infos[l], h_edge_infos[l] + old_emb_size,valid_emb, new_einfo, is_valid());
						check_cuda_error(cudaMemcpy(h_edge_infos[l], new_einfo, sizeof(label_type)*valid_emb_size, cudaMemcpyHostToHost));
					}
					//TODO we dont know whether thrust library functions works on unified memory, if not, we could use manaually reduce and copy operation by hand-write kernel functions. This is left TODO for now, and should be implemented after code review later.
					break;
		}
		check_cuda_error(cudaFree(new_vid));
		check_cuda_error(cudaFree(new_idx));
		if (enable_elabel)
			check_cuda_error(cudaFree(new_einfo));
		mem_head -= (sizes[l] - valid_emb_size); 
		sizes[l] = valid_emb_size;
		return ;
	}
	void size_adjustment() {
		emb_off_type cur_size = sizes[last_level];
		if (cur_size%32 == 0)
			return ;
		emb_off_type new_size = (cur_size + 31)/32*32;
		check_cuda_error(cudaMemset(h_vid_lists[last_level]+cur_size, -1, sizeof(KeyT)*(new_size-cur_size)));
		sizes[last_level] = new_size;
		mem_head += (new_size-cur_size);
		return ;
	}
	void remove_tail(emb_off_type idx) { sizes[last_level] = idx; }// TODO: this is a prototype of deletion operation
	void reset_level() {
		for (size_t i = 0; i <= last_level; i ++)
			sizes[i] = 0;
		last_level = 1;
		mem_head = 0;
	}
	void clean() {
		reset_level();
		free(h_vid_lists);
		free(h_idx_lists);
		check_cuda_error(cudaFree(all_vid));
		check_cuda_error(cudaFree(all_idx));
		if(enable_elabel) {
			free(h_edge_infos);
			check_cuda_error(cudaFree(all_einfo));
		}
	}
	void copy_to_level(int l, KeyT *copy_src, int dst_off, int copy_size) {
		assert(l <= last_level);
		KeyT *dst = h_vid_lists[l];
		check_cuda_error(cudaMemcpy(dst + dst_off, copy_src, sizeof(KeyT)*copy_size, cudaMemcpyDeviceToHost));
	}//for now, this is only used as the first layer of emblist.	
	__host__ __device__ inline void get_embedding(unsigned level, emb_off_type pos, KeyT *emb) {
		KeyT vid = get_vid(level, pos);
		emb[level] = vid;
		while (level > 0) {
			pos = get_idx(level, pos);
			level --;
			emb[level] = get_vid(level, pos);
		}
	}	
	
	__host__ __device__ inline void get_edge_embedding(unsigned level, emb_off_type pos, KeyT *emb, label_type *einfos) {
		KeyT vid = get_vid(level, pos);
		label_type lab = get_edge_info(level, pos);
		emb[level] = vid;
		einfos[level] = lab;
		while (level > 0) {
			pos = get_idx(level, pos);
			level --;
			emb[level] = get_vid(level, pos);
			einfos[level] = get_edge_info(level, pos);
		}
		return ;
	}
	emb_off_type check_valid_num(KeyT *emb_units, emb_off_type emb_num) {
		uint32_t block_num = 10000;
		emb_off_type *block_results;
		check_cuda_error(cudaMalloc((void **)&block_results, sizeof(emb_off_type)*BLOCK_SIZE/32*block_num));
		check_cuda_error(cudaMemset(block_results, 0, sizeof(emb_off_type)*BLOCK_SIZE/32*block_num));
		KeyT * tmp_ptr = emb_units;
		check_valid_ftr_num<<<block_num, BLOCK_SIZE>>>(tmp_ptr, block_results, emb_num);
		check_cuda_error(cudaDeviceSynchronize());
		emb_off_type valid_num = thrust::reduce(thrust::device, block_results, block_results + BLOCK_SIZE/32*block_num);
		check_cuda_error(cudaFree(block_results));
		return valid_num;
	}
	
	//TODO: have not consider when frontier residents on GPU not CPU managed memory
	//TODO: this is IMPORTANT!
	void copy_to_vid(KeyT *src, emb_off_type off, uint32_t copy_size, uint32_t l) {
		assert(mem_head + (copy_size+31)/32*32 < MAX_EMB_UNIT_NUM);
		memset(h_vid_lists[l] + copy_size/32*32, -1, sizeof(KeyT)*32);
		check_cuda_error(cudaMemcpy(h_vid_lists[l]+off, src, sizeof(KeyT)*copy_size, cudaMemcpyDeviceToHost));
		copy_size = (copy_size + 31)/32*32;
		sizes[l] += copy_size;
		mem_head += copy_size;
		return ;
	}
	void copy_to_idx(emb_off_type *src, emb_off_type off, uint32_t copy_size, uint32_t l) {
		check_cuda_error(cudaMemcpy(h_idx_lists[l]+off, src, sizeof(emb_off_type)*copy_size, cudaMemcpyDeviceToHost));
		return ;
	}
	void copy_to_vid_from_d(KeyT *src, emb_off_type off, uint32_t copy_size, uint32_t l) {
		switch(memory_type) {
			case GPU_MEM:
				printf("ERROR: COPY VID TO DEVICE EMBEDDING IS NOT SUPPORT YET!\n");
				break;
			case UNIFIED_MEM:
				assert(mem_head + (copy_size+31)/32*32 < MAX_EMB_UNIT_NUM);
				memset(h_vid_lists[last_level]+copy_size/32*32, -1, sizeof(KeyT)*32);
				check_cuda_error(cudaMemcpy(h_vid_lists[l]+off, src, sizeof(KeyT)*copy_size, cudaMemcpyDeviceToHost));
				copy_size = (copy_size + 31)/32*32;
				sizes[l] += copy_size;
				mem_head += copy_size;
				break;
		}
		return ;
	}
	void copy_to_idx_from_d(emb_off_type *src, emb_off_type off, uint32_t copy_size, uint32_t l) {
		switch(memory_type) {
			case GPU_MEM:
				printf("ERROR: COPY VID TO DEVICE EMBEDDING IS NOT SUPPORT YET!\n");
				break;
			case UNIFIED_MEM:
				memset(h_idx_lists[last_level]+copy_size/32*32, 0, sizeof(emb_off_type)*32);
				check_cuda_error(cudaMemcpy(h_idx_lists[l]+off, src, sizeof(emb_off_type)*copy_size, cudaMemcpyDeviceToHost));
				break;
		}
		return ;
	}
	void copy_to_einfo_from_d(label_type *src, emb_off_type off, uint32_t copy_size, uint32_t l) {
		switch(memory_type) {
			case GPU_MEM:
				printf("ERROR: COPY VID TO DEVICE EMBEDDING IS NOT SUPPORT YET!\n");
				break;
			case UNIFIED_MEM:
				memset(h_edge_infos[last_level]+copy_size/32*32, -1, sizeof(label_type)*32);
				check_cuda_error(cudaMemcpy(h_edge_infos[l]+off, src, sizeof(label_type)*copy_size, cudaMemcpyDeviceToHost));
				break;
		}
		return ;
	}
	 emb_off_type** get_idx_list_by_mem_controller() {return h_idx_lists;}
	 KeyT** get_vid_list_by_mem_controller() {return h_vid_lists;} 
	//KeyT **d_vid_lists;
private:
	unsigned max_level;
	unsigned last_level;
	size_t *sizes;
	//unsigned *adjacency list;
	emb_off_type** h_idx_lists;
	KeyT** h_vid_lists;
	label_type** h_edge_infos;
	emb_off_type** d_idx_lists;
	KeyT** d_vid_lists;
	label_type** d_edge_infos;
	mem_type memory_type;
	bool enable_elabel;
	//float *load_rate;//TODO: this is for later compaction opreation
};

#endif // EMBEDDING_CUH_
