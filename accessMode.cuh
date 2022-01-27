#include "graph.cuh"
#include "utils.h"
#include "embedding.cuh"
#include "expand.cuh"
__global__ void memory_page_split_identify(KeyT *memory_page_split, OffsetT *vertexList, 
								KeyT vertex_count, uint32_t total_memory_pages) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for (uint32_t i = idx; i < vertex_count; i += blockDim.x * gridDim.x) {
		long int before_pid = (i == 0) ? 10000 : (vertexList[i-1]*sizeof(KeyT)/4000);
		long int  now_pid = vertexList[i]*sizeof(KeyT)/4000;
		if (before_pid != now_pid)
			memory_page_split[now_pid] = i;
	}	
	return;
}
__global__ void set_access_mode(CSRGraph g, KeyT *memory_page_split, uint32_t total_memory_pages, 
					 uint8_t *access_mode, ATT* vertex_access_times,
					 uint64_t avg_access_per_page) {
	uint32_t warp_id = (threadIdx.x + blockIdx.x*blockDim.x)/32;
	uint32_t total_warp_num = blockDim.x * gridDim.x /32;
	uint32_t lane_id = threadIdx.x % 32;
	for (uint32_t w = warp_id; w < total_memory_pages; w += total_warp_num) {
		uint32_t cur_page_start = memory_page_split[w];
		uint32_t cur_page_end = memory_page_split[w+1];
		uint64_t valid_vertex_num = 0;
		//check page access mode
		for (uint32_t i = cur_page_start + lane_id; i < cur_page_end; i += 32) {
			uint32_t time = vertex_access_times[i];
			uint32_t deg = g.getDegree(i);
			//uint32_t threshold = (cur_page_end - cur_page_start)/20;
			//time = (time > threshold) ? threshold : time;
			//threshold = threshold > 3 ? 3 : threshold;
			//valid_vertex_num += (time > threshold) ? threshold : time;
			valid_vertex_num += (uint64_t)time*deg;
		}
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 16);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 8);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 4);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 2);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 1);
		valid_vertex_num = __shfl_sync(0xffffffff, valid_vertex_num, 0);
		uint32_t unified = 0;
		//the judgement of two access mode is to be optimized
		if (valid_vertex_num > 1.1*avg_access_per_page)
			unified = 1;
		if (!unified)
			continue;
		//set access mode bitmaps
		for (uint32_t i = cur_page_start + lane_id; i < cur_page_end; i += 32) {
			access_mode[i] = unified;
		}
	}
	return ;
}
//TODO this is an interesting problem to use GPU for elements occurance counting
//here we use atomic counting 
//instead SORT and REDUCE_BY_KEY in thrust is worthy a try
__global__ void accumulate_vertex_access_time(ATT *access_times,
											  KeyT *EL_frontier,
											  uint32_t frontier_size) {
	uint32_t idx = threadIdx.x + blockIdx.x*blockDim.x;
	for (uint32_t i = idx; i < frontier_size; i += (blockDim.x * gridDim.x)) {
		atomicAdd(access_times + EL_frontier[i], 1);
	}
	return ;
}
__global__ void count_total_access_nbr(CSRGraph g, EmbeddingList emblist, uint64_t *warp_reduced_nbr_access, 
									   emb_off_type cur_nbr_size, uint8_t level) {
	uint32_t idx = threadIdx.x + blockIdx.x* blockDim.x;
	uint32_t total_threads = blockDim.x * gridDim.x;
	uint64_t access_nbr_num = 0;
	for (emb_off_type i = idx; i < cur_nbr_size; i += total_threads) {
		KeyT cur_nbr = emblist.get_vid(level, i);
		access_nbr_num += g.getDegree(cur_nbr);
	}
	__syncthreads();
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 16);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 8);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 4);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 2);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 1);
	if (threadIdx.x % 32 == 0)
		warp_reduced_nbr_access[idx/32] = access_nbr_num;
	return ;
}
//TODO: the vertex of each layer in the embedding list determines the access mode,
//which can be saved and use later to save the recalculation time. But for now,
//We only implement the naive version.
//TODO: we should also consider the time locality of the embeddinglist -- vertices
//in the same layer of embedding list not necessarily are accessed in the same kernel
//We should also consider that later.
class access_mode_controller{
private:
	//TODO this is used for LFU-based access mode determination method
	//uint32_t avaiable_pages;
	//uint32_t page_timestamp;
	KeyT graph_total_page;
	KeyT* mem_page_vertex_border;
	uint64_t *frontier_access_neighbors;
public:
	access_mode_controller() {}
	~access_mode_controller() {}
	void set_vertex_page_border(CSRGraph g) {
		KeyT nnodes = g.get_nnodes();
		OffsetT nedges = g.get_nedges();
		graph_total_page = (sizeof(KeyT)*nedges + 3999)/4000;
		check_cuda_error(cudaMalloc((void **)&mem_page_vertex_border, sizeof(KeyT)*(graph_total_page+1)));
		memory_page_split_identify<<<20000, BLOCK_SIZE>>>(mem_page_vertex_border,
								   						 g.get_row_start_by_mem_controller(),
								   						 nnodes, graph_total_page);
		check_cuda_error(cudaMemcpy(mem_page_vertex_border+graph_total_page,
									&nnodes, sizeof(KeyT), cudaMemcpyHostToDevice));
		frontier_access_neighbors = (uint64_t *)malloc(sizeof(uint64_t)*embedding_max_length);
		memset(frontier_access_neighbors, 0, sizeof(uint64_t)*embedding_max_length);
		return ;
	}
	void cal_access_mode_by_EL(CSRGraph g, expand_constraint ec, EmbeddingList emblist){
		Clock set_mem_access("access control");
		set_mem_access.start();
		uint8_t* access_mode = g.get_access_mode_by_mem_controller();
		ATT* vertex_access_times;
		KeyT nnodes = g.get_nnodes();
		OffsetT nedges = g.get_nedges();
		check_cuda_error(cudaMemset(access_mode, 0, sizeof(uint8_t)*nnodes)); 
		check_cuda_error(cudaMalloc((void **)&vertex_access_times, sizeof(ATT)*nnodes));
		check_cuda_error(cudaMemset(vertex_access_times, 0, sizeof(ATT)*nnodes));
		
		//get total access neighbors
		uint32_t nblocks = 16000;
		uint32_t total_warps = nblocks * BLOCK_SIZE/32;
		uint64_t *warp_reduced_nbr_access;
		check_cuda_error(cudaMalloc((void **)&warp_reduced_nbr_access, sizeof(uint64_t)*total_warps));
		uint64_t total_access_neighbors = 0;
		for (uint32_t i = 0; i < ec.nbr_size; i ++) {
			uint8_t cur_nbr = (ec.nbrs>>(8*i))&0xff;
			if (frontier_access_neighbors[cur_nbr] != 0)
				total_access_neighbors += frontier_access_neighbors[cur_nbr];
			else {
				emb_off_type cur_nbr_size = emblist.size(cur_nbr);
				check_cuda_error(cudaMemset(warp_reduced_nbr_access, 0, sizeof(uint64_t)*total_warps));
				count_total_access_nbr<<<nblocks, BLOCK_SIZE>>>(g, emblist, warp_reduced_nbr_access, cur_nbr_size, cur_nbr);
				check_cuda_error(cudaDeviceSynchronize());
				frontier_access_neighbors[cur_nbr] = thrust::reduce(thrust::device, warp_reduced_nbr_access, warp_reduced_nbr_access + total_warps);
				total_access_neighbors += frontier_access_neighbors[cur_nbr];
				//printf("cur_nbr_size %d\n", cur_nbr_size);
				//printf("total access neighbors %lu\n", total_access_neighbors);
			}
		}
		check_cuda_error(cudaFree(warp_reduced_nbr_access));
		emb_off_type total_emb_size = 0;
		for (uint32_t i = 0; i < ec.nbr_size; i ++) {
			uint8_t query_vertex = (ec.nbrs>>(8*i))&0xff;
			KeyT *cur_EL_frontier = *(emblist.get_vid_list_by_mem_controller()+query_vertex);
			unsigned frontier_size = emblist.size(query_vertex);
			accumulate_vertex_access_time<<<20000, BLOCK_SIZE>>>(vertex_access_times,
																cur_EL_frontier,
																frontier_size);
			check_cuda_error(cudaDeviceSynchronize());	
			total_emb_size += frontier_size;
		}
		
		//log_info(set_mem_access.count("total access time %lu, total ftr size %d, and %d nbr access per page", total_access_neighbors, total_emb_size, total_access_neighbors/graph_total_page));
		set_access_mode<<<20000, BLOCK_SIZE>>>(g, mem_page_vertex_border,graph_total_page,
											  access_mode,vertex_access_times,
											  total_access_neighbors/graph_total_page);
		check_cuda_error(cudaDeviceSynchronize());
		log_info(set_mem_access.count("end access control"));
		//check the propotion of unified memory access;
		/*KeyT *mem_vertex_border = new KeyT [graph_total_page+1];
		check_cuda_error(cudaMemcpy(mem_vertex_border, mem_page_vertex_border, sizeof(KeyT)*(graph_total_page+1), cudaMemcpyDeviceToHost));
		uint8_t *access_mode_h = new uint8_t [nnodes];
		check_cuda_error(cudaMemcpy(access_mode_h, access_mode, sizeof(uint8_t)*nnodes, cudaMemcpyDeviceToHost));
		uint32_t unified_page_num = 0;
		for (uint32_t i = 0; i < graph_total_page; i++)
			if (access_mode_h[mem_vertex_border[i]] == 1)
				unified_page_num ++;
		log_info("the %d of %d pages use unified memory access", unified_page_num, graph_total_page);
		ofstream file;
		char buffer [10];
		sprintf(buffer, "%d", emblist.level());
		file.open(buffer);
		file << "this is the "<< emblist.level() << "th level\n";
		ATT *vertex_access_times_h = new ATT [nnodes];
		check_cuda_error(cudaMemcpy(vertex_access_times_h, vertex_access_times, sizeof(ATT)*nnodes, cudaMemcpyDeviceToHost)); 
		for (int i = 0; i < graph_total_page; i ++) {
			file << "page " << i << "'s access mode " << (uint32_t)access_mode_h[mem_vertex_border[i]] << "\n";
			uint32_t total_time = 0;
			for (int j = mem_vertex_border[i]; j < mem_vertex_border[i+1]; j ++) {
					total_time += vertex_access_times_h[j];
					file << (uint32_t)vertex_access_times_h[j] << "\n";
			}
			file << "total page " << mem_vertex_border[i+1] - mem_vertex_border[i] <<
					" total access time " << total_time << "\n";
		}
		file.close();
		log_info(set_mem_access.count("end access mem control check"));
		delete [] mem_vertex_border;
		delete [] access_mode_h;
		delete [] vertex_access_times_h;*/
		check_cuda_error(cudaFree(vertex_access_times));
	}
	void clean() {
		check_cuda_error(cudaFree(mem_page_vertex_border));
		free(frontier_access_neighbors);
		return ;
	}
};
