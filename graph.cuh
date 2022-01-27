#pragma once

#ifndef GRAPH_H
#define GRAPH_H
struct Edge {
  Edge(KeyT from, KeyT to) : src(from), dst(to) {}
  KeyT src;
  KeyT dst;
};

class CSRGraph {
public:
  OffsetT* row_start;
  KeyT* edge_dst;
  //OffsetT* row_start_zc;
  KeyT* edge_dst_zc;
  KeyT *edge_dst_h;
  node_data_type* node_data;
/*#ifdef ENABLE_ELABEL
  edge_data_type* edge_label;
  edge_data_type* edge_label_zc;
  edge_data_type* edge_label_h;
#endif*/
  uint32_t nnodes;
  uint64_t nedges;
  //bool need_dag;
  bool use_node_data;
/*#ifdef ENABLE_ELABEL
  bool use_edge_data;
#endif */
  mem_type memory_type;
  uint8_t* access_mode;//TODO: we can also use bitmap here to record access mode


public:
  CSRGraph() { init(); }
  //~CSRGraph() {}
  void init() {
    row_start = NULL;
    edge_dst = edge_dst_zc = edge_dst_h = NULL;
/*#ifdef ENABLE_ELABEL
    edge_label = edge_label_zc = edge_label_h = NULL;
    use_edge_data   = false;
#endif
*/
    node_data            = NULL;
    nnodes = nedges = 0;
    //need_dag        = false;
    use_node_data   = false;
    access_mode = NULL;//Note: 0 for zero copy memory, 1 for unified memory
  }
  //void enable_dag() { need_dag = true; }
  __device__ __host__ KeyT get_nnodes() { return nnodes; }
  __device__ __host__ OffsetT get_nedges() { return nedges; }
  __host__ __device__ uint8_t access_mode_of(KeyT node){
      return access_mode[node];
  }
  void clean() {
    check_cuda_error(cudaFree(row_start));
    if (edge_dst != NULL) check_cuda_error(cudaFree(edge_dst));
    //check_cuda(cudaFreeHost(row_start_zc));
    if (node_data != NULL) check_cuda_error(cudaFree(node_data));
/*#ifdef ENABLE_ELABEL
    if (edge_label != NULL) check_cuda_error(cudaFree(edge_label));
    if (edge_label_h != NULL) free(edge_label_h);
#endif*/
    if (edge_dst_h != NULL) check_cuda_error(cudaFreeHost(edge_dst_h));
    check_cuda_error(cudaFree(access_mode));
  }
  __device__ __host__ OffsetT* get_row_start_by_mem_controller() {return row_start;}
  __device__ __host__ uint8_t* get_access_mode_by_mem_controller() {return access_mode;}
  __device__ __host__ bool valid_node(KeyT node) { return (node < nnodes); }
  __device__ __host__ bool valid_edge(KeyT edge) { return (edge < nedges); }
  __device__ __host__ KeyT getDegree(KeyT src) {//TODO why this function can be called from both host and device ?
    if (src == 0xffffffff)
	return 0;
    assert(src < nnodes);
    return  row_start[src + 1] - row_start[src];
  };
  __device__ __host__ KeyT getDestination(unsigned src, unsigned edge) {
    assert(src < nnodes);
    assert(edge < getDegree(src));
    OffsetT abs_edge = row_start[src] + edge;
    assert(abs_edge < nedges);
    return access_mode_of(src) ? edge_dst[abs_edge] : edge_dst_zc[abs_edge];
  };
  __device__ __host__ KeyT getAbsDestination(OffsetT abs_edge) {
    assert(abs_edge < nedges);
    return edge_dst[abs_edge];//TODO: zero copy for now, but should be determined by cases
  };
  inline __device__ __host__ KeyT getEdgeDst(OffsetT edge) {
    assert(edge < nedges);
    return edge_dst[edge];
  };
  inline __device__ __host__ KeyT getEdgeDstOfSrc(OffsetT edge, KeyT src) {
    return access_mode_of(src) ? edge_dst[edge] : edge_dst_zc[edge];
  };
  inline __device__ __host__ KeyT* getAdjListofSrc(OffsetT src, OffsetT off) {
    return access_mode_of(src) ? edge_dst + off: edge_dst_zc + off;
  };
  inline __device__ __host__ node_data_type getData(unsigned vid) {
    return node_data[vid];
  };
  inline __device__ __host__ OffsetT edge_begin(unsigned src) {
    if (src == 0xffffffff)
	return 0;
    assert(src < nnodes);
    return  row_start[src];
  };
  inline __device__ __host__ OffsetT edge_end(unsigned src) {
    if (src == 0xffffffff)
	return 0;
    assert(src < nnodes);
    return  row_start[src + 1];
  };
  void read(std::string file, bool read_labels, mem_type mem_tp) {
    std::cout << "Reading graph from file: " << file << "\n";
    //need_dag = dag;
    memory_type = mem_tp;
    read_topology_graph(file.c_str());
    check_cuda_error(cudaMalloc((void **)&access_mode, sizeof(uint8_t)*nnodes));
	switch (memory_type) {
		case GPU_MEM:
    		check_cuda_error(cudaMemset(access_mode, -1, sizeof(uint8_t)*nnodes));
			break;
		case UNIFIED_MEM:
    		check_cuda_error(cudaMemset(access_mode, -1, sizeof(uint8_t)*nnodes));
			break;
		case ZERO_COPY_MEM:
    		check_cuda_error(cudaMemset(access_mode, 0, sizeof(uint8_t)*nnodes));
			break;
		case COMBINED_MEM://TODO I think set to zero is better
    		check_cuda_error(cudaMemset(access_mode, -1, sizeof(uint8_t)*nnodes));
			break;
	}
    if (read_labels) {
      use_node_data = true;
/*#ifdef ENABLE_ELABEL
      use_edge_data = true;
#endif*/
      read_graph_labels(file.c_str());
    } else {
      use_node_data = false;
    }
    return ;
  }
  void read_graph_labels(const char file_name[]){
    std::string fileneme = file_name;
    std::string vertex_label_file = file_name + std::string(".vlabel");
	//if define and use edge label
    //std::string edge_label_file = file_name + ".elabel";
    std::ifstream file;
    file.open(vertex_label_file.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()){
       fprintf(stderr, "vertex label file open failed\n");
       exit(1);
    }
    file.read((char*)&nnodes, sizeof(uint32_t));
    node_data_type *node_data_h = (node_data_type *)malloc(nnodes*sizeof(node_data_type));
    file.read((char*)node_data_h, sizeof(node_data_type)*nnodes);
    check_cuda_error(cudaMalloc((void **)&node_data, sizeof(node_data_type)*nnodes));
    check_cuda_error(cudaMemcpy(node_data, node_data_h, sizeof(node_data_type)*nnodes, cudaMemcpyHostToDevice));
    check_cuda_error(cudaDeviceSynchronize());
    free(node_data_h);
    node_data_h = NULL;
    file.close();

	//if define and use edge label
/*#ifdef ENABLE_ELABEL
    file.open(edge_label_file.c_str(), std::ios::in | std::ios::bianry);
    if (!file.is_open()){
        fprintf(stderr, "edge label file open failed\n");
        exit(1);
    }
    file.read((char*)&nedges, 8);
    switch (mem) {
        case GPUMEM:
            edge_label_h = (edge_data_type *)malloc(nedges*sizeof(edge_data_type));
            //edge_label_h = (edge_data_type*)malloc(sizeof(edge_data_type)*nedges);
            file.read((char*)edge_label_h, nedges*sizeof(edge_data_type));
            //file2.read((char*)edge_label_h, nedges*sizeof(edge_data_type));
            check_cuda_error(cudaMalloc((void**)&edge_label, nedges*sizeof(edge_data_type)));
            check_cuda_error(cudaDeviceSynchronize());
            free(edge_label_h);
            edge_label_h = NULL;
            //check_cuda_error(cudaMalloc((void**)&weightList_d, weight_size));
            break;
        case UNIFIED_MEM:
            check_cuda_error(cudaMallocManaged((void**)&edge_label, nedges*sizeof(edge_data_type)));
            //check_cuda_error(cudaMallocManaged((void**)&weightList_d, weight_size));
            file.read((char*)edge_label, nedges*sizeof(edge_data_type));
            //file2.read((char*)weightList_d, weight_size);
            check_cuda_error(cudaMemAdvise(edge_label, nedges*sizeof(edge_data_type), cudaMemAdviseSetReadMostly, device));
            check_cuda_error(cudaDeviceSynchronize());
            //check_cuda_error(cudaMemAdvise(edgeList_d, nedges, cudaMemAdviseSetAccessedBy, device));
            break;
        case ZERO_COPY_MEM:
            check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
            check_cuda_error(cudaHostAlloc(&edge_label_h, sizeof(edge_data_type)*nedges, cudahostAllocMapped));
            file.read((char *)edge_label_h, nedges*sizeof(edge_data_type));
            check_cuda_error(cudaHostGetDevicePointer((void **)edge_label_zc, (void **)edge_label_h, 0));
            
            break;
        case COMBINED_MEM:
            check_cuda_error(cudaMallocManaged((void **)&edge_label, nedges*sizeof(edge_data_type)));
            file.read((char *)edge_label, nedges*sizeof(edge_data_type));

            check_cuda_error(cudaMemAdvise(edge_label, nedges*sizeof(edge_data_type), cudaMemAdviceSetAccessBy, device));

            check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
            check_cuda_error(cudaHostAlloc(&edge_label_h, sizeof(edge_data_type)*nedges, cudaHostAllocMapped));
            memcpy(edge_label_h, edge_label, sizeof(edge_data_type)*nedges);
            check_cuda_error(cudaHostGetDevicePointer((void **)edge_label_zc, (void **)edge_label_h, 0));
            break;
    }

    file.close();
#endif*/
    return;
  }
  void read_topology_graph(const char file_name[]) { 
    std::string fileneme = file_name;
    std::string vertex_file = file_name + std::string(".col");
    std::string edge_file = file_name + std::string(".dst");
    //std::string weight_file = filename + ".val";

    std::ifstream file;
    file.open(vertex_file.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Vertex file open failed\n");
        exit(1);
    }
    //read row_off
    
    file.read((char*)(&nnodes), 4);
    //file.read((char*)(&typeT_64), 8);

    uint64_t *row_start_h = (uint64_t *)malloc((nnodes+1)*sizeof(uint64_t));
    file.read((char *)row_start_h, sizeof(uint64_t)*(nnodes+1));
    file.close();
    check_cuda_error(cudaMalloc((void **)&row_start, sizeof(OffsetT)*(nnodes+1)));
    check_cuda_error(cudaMemcpy(row_start, row_start_h, sizeof(OffsetT)*(nnodes+1), cudaMemcpyHostToDevice));
    check_cuda_error(cudaDeviceSynchronize());
    free(row_start_h);
    //read edge_dst and edge weight
    file.open(edge_file.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Edge file open failed\n");
        exit(1);
    }

    file.read((char*)(&nedges), 8);
    //file.read((char*)(&typeT), 8);
    //edge_data_type *edge_label_h;

    switch (memory_type) {
        case GPU_MEM:
            edge_dst_h = (uint32_t *)malloc(nedges*sizeof(uint32_t));
            //edge_label_h = (edge_data_type*)malloc(sizeof(edge_data_type)*nedges);
            file.read((char*)edge_dst_h, nedges*sizeof(uint32_t));
            //file2.read((char*)edge_label_h, nedges*sizeof(edge_data_type));
            check_cuda_error(cudaMalloc((void**)&edge_dst, nedges*sizeof(uint32_t)));
            check_cuda_error(cudaMemcpy(edge_dst, edge_dst_h, sizeof(uint32_t)*nedges, cudaMemcpyHostToDevice));
            check_cuda_error(cudaDeviceSynchronize());
            free(edge_dst_h);
            edge_dst_h = NULL;
            //check_cuda_error(cudaMalloc((void**)&weightList_d, weight_size));
            /*for (uint64_t i = 0; i < weight_count; i++)
                edge_label_h[i] += offset;
            */
            break;
        case UNIFIED_MEM:
            check_cuda_error(cudaMallocManaged((void**)&edge_dst, nedges*sizeof(uint32_t)));
            //check_cuda_error(cudaMallocManaged((void**)&weightList_d, weight_size));
            file.read((char*)edge_dst, nedges*sizeof(uint32_t));
            //file2.read((char*)weightList_d, weight_size);
			//TODO
	    //best performance
            check_cuda_error(cudaMemAdvise(edge_dst, nedges*sizeof(uint32_t), cudaMemAdviseSetReadMostly, 0));
            //worst performance
	    //check_cuda_error(cudaMemAdvise(edge_dst, nedges*sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
            check_cuda_error(cudaDeviceSynchronize());
	    //also bad
            //check_cuda_error(cudaMemAdvise(edge_dst, nedges*sizeof(uint32_t), cudaMemAdviseSetAccessedBy, 0));
            break;
        case ZERO_COPY_MEM:
            //check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
            check_cuda_error(cudaHostAlloc(&edge_dst_h, sizeof(uint32_t)*nedges, cudaHostAllocMapped));
            file.read((char *)edge_dst_h, nedges*sizeof(uint32_t));
            check_cuda_error(cudaHostGetDevicePointer((void **)&edge_dst_zc, (void *)edge_dst_h, 0));
            
            break;
        case COMBINED_MEM:
	    check_cuda_error(cudaMallocManaged((void**)&edge_dst, nedges*sizeof(uint32_t)));
            //check_cuda_error(cudaMallocManaged((void**)&weightList_d, weight_size));
            file.read((char*)edge_dst, nedges*sizeof(uint32_t));
            //file2.read((char*)weightList_d, weight_size);
			//TODO
            check_cuda_error(cudaMemAdvise(edge_dst, nedges*sizeof(uint32_t), cudaMemAdviseSetReadMostly, 0));
            check_cuda_error(cudaDeviceSynchronize());

            //check_cuda_error(cudaMemAdvise(edge_dst, nedges*sizeof(uint32_t), cudaMemAdviseSetAccessedBy, 0));

            check_cuda_error(cudaHostAlloc(&edge_dst_h, sizeof(uint32_t)*nedges, cudaHostAllocMapped));
            memcpy(edge_dst_h, edge_dst, sizeof(uint32_t)*nedges);
            check_cuda_error(cudaHostGetDevicePointer((void **)&edge_dst_zc, (void *)edge_dst_h, 0));
            break;
    }

    file.close();
    //file2.close();
    return ;
  }
 };
#endif
