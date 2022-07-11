# GAMMA
A graph pattern mining framework for large graphs on gpu.

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
