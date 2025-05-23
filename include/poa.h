#ifndef POA_H
#define POA_H

#include <unordered_map>
#include <vector>
#include <random>
#include "task.h"
#include "gfaToGraph.h"

typedef short Edge;

using namespace std;

void read_batch(vector<vector<string>> &batch, size_t size, string filename);

void get_bmean_batch_result_gpu(vector<vector<string>> windows, int &c, const int numBlocks, int batchSize, graph_h* g);

void print_reads(vector<vector<string>> reads);

int check_input_int(string &arg);

void checkCudaDeviceLimits(int seqLen, int shared_size, long long global_size);

__global__ void printGraphStructure(int numBlocks, int batchStructure);

__global__ void print_lpo_offsets(int numBlocks, int batchStructure);

__global__ void assign_device_memory(char* dletters, Edge* dedges, int* dedgebounds, unsigned char* moves, short* diagonals_sc, 
                                    int* d_offs, int* xy, int* yx, int* dynlg, const int num_blocks, int* roffs, char* reads, int* nE, int seqLen, int* res);

 __global__ void init_diagonals(int i_seq_idx, int j_seq_idx, int uses_global, int* nseq_offsets);

 __global__ void sw_align(int i_seq_idx, int j_seq_idx, int uses_global, int* nseq_offsets); 
	
 __device__ void trace_back_lpo_alignment(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, 
                                        int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

 __global__ void compute_d_offsets(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

 __global__ void compute_edge_offsets(int* seq_offsets, int* nseq_offsets);

 __global__ void generate_lpo(char* seq, int* seq_offsets, int* nseq_offsets);

 #endif