#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "accessMode.cuh"
#include "expand.cuh"
#include "queryGraph.cuh"
#include <cuda_runtime.h>

using namespace std;
__global__ void set_validation(CSRGraph g, uint8_t *valid_candi, uint32_t nnodes, uint8_t lab, uint32_t min_deg) {
	uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
	for (uint32_t i = tid; i < nnodes; i += (blockDim.x*gridDim.x)) {
		if (g.getDegree(i) >= min_deg && g.getData(i) == lab)
			valid_candi[i] = 1;
	}
	return ;
}
int main(int argc, char *argv[]) {
	if (argc < 4) {
		printf("usage: ./sm ($data_graph) ($query_graph) graph_mt debug\n");
		return 0;
	}
	if (string(argv[argc-1]) != "debug") {
		log_set_quiet(true);
	}
	Clock start("Start");
	//assert(k <= embedding_max_length);
	queryGraph Gquery;
	Gquery.readFromFile(argv[2]);
	std::string file_name = argv[1];
	CSRGraph data_graph;
	mem_type mt_emb = (mem_type)1;//0 GPU 1 Unified 2 Zero 3 Combine
	mem_type mt_graph = (mem_type)atoi(argv[3]);
	if (mt_graph > 1)
		check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
	data_graph.read(file_name, true, mt_graph);//for sm. we should definitely read vertex labels
	log_info(start.start());
	log_info(start.count("nedges %lu, nnodes %d", data_graph.get_nedges(), data_graph.get_nnodes()));
	EmbeddingList emb_list;
	uint32_t nnodes = data_graph.get_nnodes();
	uint64_t nedges = data_graph.get_nedges();

	uint8_t query_node = Gquery.core[0];
	uint8_t q_label = Gquery.vertex_label[query_node];
	uint32_t q_deg = Gquery.adj_list[query_node].size();
	
	KeyT *seq, *results; 
	cudaMalloc((void **)&seq, sizeof(KeyT)*nnodes);
	check_cuda_error(cudaMalloc((void **)&results, sizeof(KeyT)*nnodes));
	check_cuda_error(cudaMemset(results, -1, sizeof(KeyT)*nnodes));
	uint8_t *valid_candi;
	check_cuda_error(cudaMalloc((void **)&valid_candi, sizeof(uint8_t)*nnodes));
	check_cuda_error(cudaMemset(valid_candi, 0, sizeof(uint8_t)*nnodes));
	set_validation<<<10000, 256>>>(data_graph, valid_candi, nnodes, q_label, q_deg);
	thrust::sequence(thrust::device, seq, seq + nnodes);
	uint32_t valid_node_num = thrust::copy_if(thrust::device, seq, seq + nnodes, valid_candi, results, is_valid())- results;
	check_cuda_error(cudaDeviceSynchronize());
	emb_list.init(valid_node_num, Gquery.core.size(), mt_emb, false);
	emb_list.copy_to_level(0, results, 0, valid_node_num);
	check_cuda_error(cudaFree(seq));
	check_cuda_error(cudaFree(valid_candi));
	check_cuda_error(cudaFree(results));


	//emb_list.init(nnodes, Gquery.core.size(), mt_emb, false);//TODO: need better initialization strategy: take degree and vertex label into consideration
	log_info(start.count("embedding initialization done!"));
	//TODO: here we plan to add a optimizer to determine expand order, expand constraint, and so on.
	//set the first level
	//KeyT *seq = (KeyT*)malloc(sizeof(KeyT)*nnodes);
	//thrust::sequence(thrust::host, seq, seq + nnodes);
	//emb_list.copy_to_level(0, seq, 0, nnodes);
	//free(seq);
	//log_info("the valid num of layer 0 is %lu",emb_list.check_valid_num(0));
	//set the second level
	//emb_list.add_level(nedges);
	//expand for every vertex in the query graph
	access_mode_controller access_controller;
	access_controller.set_vertex_page_border(data_graph);
	log_info(start.count("access controller initalization done!"));
	Clock Expand("Expand");
	log_info(Expand.start());
	/*for (uint32_t i = 0; i < Gquery.vertex_num; i ++) {
		cout << "here is vertex " << i << ":" << endl;
		cout << "nbr ";
		for (uint32_t j = 0; j < Gquery.adj_list[i].size(); j ++) {
			cout << Gquery.adj_list[i][j] << " ";
		}
		cout << "order nbr:";
		for (uint32_t j = 0; j < Gquery.order_list[i].size(); j ++) {
			cout << Gquery.order_list[i][j] << " ";
		}
		cout << endl;
	}
	for (uint32_t i = 0; i < Gquery.core.size(); i ++)
		cout << (uint32_t)Gquery.core[i] << " ";
	cout << endl;
	for (uint32_t i = 0; i < Gquery.satellite.size(); i ++)
		cout << (uint32_t)Gquery.satellite[i] << " ";
	cout << endl;*/

	for (int i = 1; i < Gquery.core.size(); i ++) {
		//construct the expand constraint
		uint64_t _nbrs = 0, _order_nbr = 0;
		//int8_t *_order_nbr_cmp = new int8_t [i];
		uint32_t query_node = Gquery.core[i];
		//cout << "the query node now is " << query_node << endl;
		uint32_t e_nbr_size = 0, e_order_nbr_size = 0;
		for (uint32_t j = 0; j < Gquery.adj_list[query_node].size(); j ++) {
			uint32_t cur_nbr = Gquery.adj_list[query_node][j];
			//cout << query_node << " " << cur_nbr << " " ; cout << Gquery.adj_list[query_node][j];
			if (query_node > Gquery.adj_list[query_node][j]) {//here we use id as matching order, resulting in this
				_nbrs = _nbrs | ((cur_nbr&0xff) << (j*8));
				e_nbr_size ++;
			}
			//cout  << " " << e_nbr_size << endl;
		}
		for (uint32_t j = 0; j < Gquery.order_list[query_node].size(); j ++) {
			if (query_node > Gquery.order_list[query_node][j]) {
				_order_nbr = _order_nbr | ((Gquery.order_list[query_node][j]&0xff) << (j*8));
				e_order_nbr_size ++;
			}
		}
		expand_constraint ec((node_data_type)Gquery.vertex_label[query_node], Gquery.adj_list[query_node].size(),
							 _nbrs, e_nbr_size, 
							 (emb_order)1, _order_nbr, e_order_nbr_size);
		//cout << "adjacency list and order list length is " << e_nbr_size << " " << e_order_nbr_size << endl;
		//expand
		log_info(Expand.count("for the %dth iteration, start expand... ...",i));
		bool write_back = (i == Gquery.core.size()-1) ? false : true;
		expand_dynamic(data_graph, emb_list, i, ec, write_back);
		//expand_in_batch(data_graph, emb_list, i, ec);
		log_info(Expand.count("for the %dth iteration, end expand",i));
		//log_info("the valid num of layer %d is %lu", i, results);
		//set access mode
		if (mt_graph == 3) {
			Expand.pause();
			access_controller.cal_access_mode_by_EL(data_graph, ec, emb_list);
			Expand.goon();
		}
		log_info(Expand.count("for the %dth iteration, end set access mode",i));
		//delete ec;
	}
	log_info(Expand.count("end expand"));
	log_info(start.count("k-clique count ends."));
	//#TODO copy the results back to CPU and check the results;
	//CSRGraph data_graph_h;
	//data_graph.copy_to_cpu(data_graph_h);
	//#show the results in data_graph_h
	//printf("the numbers of the matched subgraph is %lu\n", emb_list.size(Gquery.core.size()-1));
	emb_list.clean();
	access_controller.clean();
	data_graph.clean();
	return 0;
}
	
