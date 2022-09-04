# GAMMA
A graph pattern mining framework for large graphs on gpu. This is the source code of "GAMMA: A Graph Pattern Mining Framework for Large Graphs on GPU" (ICDE 2023).

ABSTRACT

Graph pattern mining (GPM) is getting increasingly important recently. There are many parallel frameworks for GPM, many of which suffer from performance. GPU is a power- ful option for graph processing, which has excellent potential for performance improvement; however, parallel GPM algorithms produce a large number of intermediate results, limiting GPM implementations on GPU.

In this paper, we present GAMMA, an out-of-core GPM framework on GPU, and it makes full use of host memory to process large graphs. Specifically, GAMMA adopts a self- adaptive implicit host memory access manner to achieve high bandwidth, which is transparent to users. GAMMA provides flexible and effective interfaces for users to build their algorithms. We also propose several optimizations over primitives provided by GAMMA in the out-of-core GPU system. Experimental results show that GAMMA has scalability advantages in graph size over the state-of-the-art by an order of magnitude, and is also faster than existing GPM systems and some dedicated GPU algorithms of specific graph mining problems.

Please cite our paper, if you use our source code.

REQUIREMENTS

- GCC 5.3.0
- CUDA toolkit 9.0

INPUTS

GAMMA use CSR (compressed sparse rows) as the data structure of graphs. It has two parts: **OFFSET** and **VALUES**. 
The former records the offsets of all adjacency lists in CSR, the latter is the set of all adjacency lists.

The input file of a dataset has two files, **data.col** and **data.dst**, which are both binary files. 
**data.col** starts with the vertex number of input graph, then follows the **OFFSET**. So it is as:

```
${vertex_number} (4 bytes)
${offset_1} (8 bytes)
${offset_2} (8 bytes)
${offset_3} (8 bytes)
… …
```
**data.dst** starts with the edge number of input graph, then follows the **VALUES**. So it is as:
```{edge_number} (8 byte)
${value_1} (4_byte)
${value_2} (4_byte)
${value_3} (4_byte)
… …
```

BUILD

```
$ cd ${GAMMA_ROOT}
$ make
```
RUN

K-Clique counting.

```
$ ./kcl ${data} $k debug
```

Subgraph matching.

```
./sm ${data} ${query_name} debug
```

Frequent pattern mining.

```
./fsm ${data} ${pattern_length} ${minimum_support} debug
```
