#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "accessMode.cuh"
#include "expand.cuh"
#include "aggregrate.cuh"
#include <cuda_runtime.h>
using namespace std;
int main(int argc, char *argv[]) {
	if (argc < 5) {
		printf("usage: ./fsm ($data_graph) $(pattern_size) $(minimum_support) $(graph_mem) debug\n");
		return 0;
	}
	if (string(argv[argc-1]) != "debug") {
		log_set_quiet(true);
	}
	Clock start("Start");
	//assert(k <= embedding_max_length);
	std::string file_name = argv[1];
	CSRGraph data_graph;
	mem_type mt_emb = (mem_type)1;//0 GPU 1 Unified 2 Zero 3 Combine
	mem_type mt_graph = (mem_type)atoi(argv[4]);
	if (mt_graph > 1)
		check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
	data_graph.read(file_name, true, mt_graph);//for fsm we should definitely read vertex labels
	log_info(start.start());
	log_info(start.count("nedges %lu, nnodes %d", data_graph.get_nedges(), data_graph.get_nnodes()));
	EmbeddingList emb_list;
	uint32_t nnodes = data_graph.get_nnodes();
	uint64_t nedges = data_graph.get_nedges();
	int pattern_size = atoi(argv[2]);
	emb_list.init(nnodes, pattern_size, mt_emb, true);//TODO: need better initialization strategy: take degree and vertex label into consideration
	log_info(start.count("embedding initialization done!"));
	//check_cuda_error(cudaDeviceSynchronize());
	//TODO: here we plan to add a optimizer to determine expand order, expand constraint, and so on.
	//set the first level
	KeyT *seq;
	check_cuda_error(cudaMalloc((void **)&seq,sizeof(KeyT)*nnodes));
	thrust::sequence(thrust::device, seq, seq + nnodes);
	emb_list.copy_to_level(0, seq, 0, nnodes);
	check_cuda_error(cudaFree(seq));
	printf("after embedding initialization\n");
	//emb_list.display(0, 100);
	//log_info("the valid num of layer 0 is %lu",emb_list.check_valid_num(0));
	//set the second level
	//emb_list.add_level(nedges);
	//expand for every vertex in the query graph
	access_mode_controller access_controller;
	access_controller.set_vertex_page_border(data_graph);
	log_info(start.count("access controller initalization done!"));
	Clock fsm("FSM");
	log_info(fsm.start());
	int min_sup = atoi(argv[3]);
	uint32_t *freq_edge_patterns;
	check_cuda_error(cudaMalloc((void **)&freq_edge_patterns, sizeof(uint32_t)*max_label*max_label/32));
	check_cuda_error(cudaMemset(freq_edge_patterns, -1, sizeof(uint32_t)*max_label*max_label/32));
	for (int i = 1; i < pattern_size; i ++) {
		//expand
		log_info(fsm.count("for the %dth iteration, start expand... ...",i));
		expand_dynamic_by_edge(data_graph, emb_list, i, freq_edge_patterns);
		log_info(fsm.count("for the %dth iteration, end expand",i));
		//log_info("check valid num : %lu", emb_list.check_valid_num(i));	
		//printf("after expand:\n");
		//emb_list.display_valid(i, emb_list.size());
		printf("before compaction, the emb size is %lu\n", emb_list.size());
		emb_compaction(emb_list, i);
		log_info("now the size of embedding list is %lu", emb_list.size());
		//printf("after compaction\n");
		//emb_list.display(i, emb_list.size());
		//map embeddings to pattern id
		struct patternID *pattern_ids;
		emb_off_type emb_nums = emb_list.size(i);
		check_cuda_error(cudaMalloc((void **)&pattern_ids, sizeof(patternID)*emb_nums));//TODO: out of mem?
		check_cuda_error(cudaMemset(pattern_ids, -1, sizeof(patternID)*emb_nums));//TODO: reasonable?
		map_embeddings_to_pids(data_graph, emb_list, pattern_ids, i);
		log_info("for the %dth iteration, map embedding to pids finished.", i);
		//aggregate pattern id and obtain support ,filter valid embeddings
		aggregrate_and_filter(data_graph, emb_list, pattern_ids, i, atoi(argv[3]), freq_edge_patterns);
		emb_off_type results = emb_list.size();
		log_info("the valid num of layer %d is %lu", i, results);
		//set access mode
		//TODO access controller enabled later
		//if (mt_graph == 3) {
		//	fsm.pause();
		//	access_controller.cal_access_mode_by_EL(data_graph, ec, emb_list);
		//	fsm.goon();
		//}
		log_info(fsm.count("for the %dth iteration, end set access mode",i));
	}
	check_cuda_error(cudaFree(freq_edge_patterns));
	log_info(fsm.count("end expand"));
	log_info(start.count("frequent pattern mining process ends."));
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
	
