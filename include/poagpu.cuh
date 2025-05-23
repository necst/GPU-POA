#ifndef POAGPU_H
#define POAGPU_H

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <unistd.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <numeric>
#include <stdexcept>
#include <thread>
#include "poa.h"

#define DEBUG 1

using namespace std;
using namespace chrono;
using namespace poa_gpu_utils;

#define NOW high_resolution_clock::now() 

#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {

	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void checkCudaDeviceLimits(int seqLen, int shared_size, long long global_size) {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

	std::cout << "\nCUDA device properties:\n";
    std::cout << "Checking CUDA device launch parameters...\n";
    std::cout << "Device name: " << prop.name << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max shared memory per block (bytes): " << prop.sharedMemPerBlock << "\n";
	std::cout << "Global memory size (MegaBytes): " << prop.totalGlobalMem / (1024 * 1024) << "\n";

    if (seqLen + 1 > prop.maxThreadsPerBlock) {
        std::cerr << "ERROR: seqLen + 1 (" << (seqLen + 1)
                  << ") exceeds maxThreadsPerBlock (" << prop.maxThreadsPerBlock << ")\n";
				  abort();
    } else {
        std::cout << "OK: Thread count per block is within limit.\n";
    }

    if (shared_size > prop.sharedMemPerBlock) {
        std::cerr << "ERROR: Requested shared memory (" << shared_size
                  << " bytes) exceeds device limit (" << prop.sharedMemPerBlock << " bytes)\n";
				  abort();
    } else {
        std::cout << "OK: Shared memory used is: " << shared_size << " bytes, which is within limit.\n";
    }

	if (global_size > prop.totalGlobalMem) {
        std::cerr << "ERROR: Requested global memory (" << global_size
                  << " bytes) exceeds device limit (" << prop.totalGlobalMem << " bytes)\n";
				  abort();
    } else {
        std::cout << "OK: Global memory used is: " << global_size / (1024 * 1024) << " MegaBytes, which is within limit.\n";
    }

    std::cout << "Check complete.\n\n";
}


inline void gpu_POA_alloc(TaskRefs &T, const int numBlocks, int batchSize, int nV, int nE, size_t seqLen){

	// alloco memoria sul device, la memoria è puntata da puntatori che risiedono sulla memoria dell'host

	cudaErrchk(cudaMalloc(&T.space_exceeded, sizeof(int)));

	cudaErrchk(cudaMalloc(&T.dyn_len_global_d, (unsigned long long)numBlocks * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.dyn_nE_global_d, (unsigned long long)numBlocks * sizeof(int)));

	cudaErrchk(cudaMalloc(&T.sequences_d, (unsigned long long)seqLen * batchSize * numBlocks)); 
	cudaErrchk(cudaMalloc(&T.seq_offsets_d, (unsigned long long)numBlocks * batchSize * sizeof(int))); 
	cudaErrchk(cudaMalloc(&T.nseq_offsets_d, (unsigned long long)numBlocks * sizeof(int))); 

	cudaErrchk(cudaMalloc(&T.dyn_letters_global_d, (unsigned long long)nV * numBlocks));
	cudaErrchk(cudaMalloc(&T.dyn_edges_global_d, (unsigned long long)nE * numBlocks * sizeof(Edge)));
	cudaErrchk(cudaMalloc(&T.dyn_edge_bounds_global_d, (unsigned long long)(nV+1) * numBlocks * sizeof(int)));
	
	cudaErrchk(cudaMalloc(&T.moves_global_d, (unsigned long long)2 * (nV+1) * (seqLen+1) * numBlocks * sizeof(unsigned char)));
	cudaErrchk(cudaMalloc(&T.diagonals_global_sc_d, (unsigned long long)(nV+1)*(seqLen+1) * numBlocks * sizeof(short)));

	cudaErrchk(cudaMalloc(&T.d_offsets_global_d, (unsigned long long)(nV + seqLen+1) * numBlocks * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.x_to_ys_d, (unsigned long long)nV * numBlocks * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.y_to_xs_d, (unsigned long long)nV * numBlocks * sizeof(int)));

	cudaErrchk(cudaMalloc(&T.results_d, (unsigned long long)numBlocks * sizeof(int)));
}

inline void gpu_POA_free(TaskRefs &T){

	cudaDeviceReset();

}

static const int NOT_ALIGNED = -1;

void init_kernel_block_parameters(vector<vector<string>> &reads, char** sequences, vector<int> &nseq_offsets, 
								vector<int> &seq_offsets, int* tot_nseq, int first_el, const int numBlocks, int batchSize) {
						
	int n = (first_el + numBlocks < reads.size()) ? numBlocks : reads.size() - first_el;

	// filling nseq_offset
	for(int window_idx = first_el; window_idx < first_el + numBlocks && window_idx < reads.size(); window_idx++) {
		
		nseq_offsets[window_idx - first_el] = reads[window_idx].size();
	}
	partial_sum(nseq_offsets.begin(),nseq_offsets.end(),nseq_offsets.begin());

	*tot_nseq = nseq_offsets[n-1];
	seq_offsets = vector<int>(*tot_nseq);

	int sequence_idx = 0;
	for(int window_idx = first_el; window_idx < first_el + numBlocks && window_idx < reads.size(); window_idx++) {
		vector<string> &window = reads[window_idx];			
		int wsize = window.size();
		for(int i = 0; i < wsize; i++) {
			seq_offsets[sequence_idx] = window[i].size();
			sequence_idx++;
		}
	}
	partial_sum(seq_offsets.begin(), seq_offsets.end(), seq_offsets.begin());

	int tot_size = seq_offsets[sequence_idx-1];

	*sequences = (char*)malloc(tot_size);
	sequence_idx = 0;

	for(int window_idx = first_el; window_idx < first_el + numBlocks && window_idx < reads.size(); window_idx++) {
			
		vector<string>& window = reads[window_idx];	
		for(auto seq : window) {
			int offset;
			if(sequence_idx == 0){
				offset = 0;
			}else{
				offset = seq_offsets[sequence_idx-1];
			}
			char* seq_ptr = (*sequences) + offset;
			memcpy(seq_ptr, seq.c_str(), seq.size());
			sequence_idx++;
		}
	}	
}


void gpu_POA(vector<vector<string>> &reads, TaskRefs &T, const int numBlocks, int batchSize, graph_h* g, size_t seqLen) {

	auto start = NOW;

	int nV = g->dyn_len_global;
	int nE = g->edgesNumber;

	int numReads = 0;
	for(int i = 0; i < reads.size(); i++) {
		numReads += reads[i].size();
	}

	T.results = (int*)malloc(numReads * sizeof(int));

	int lastBatch = numReads % numBlocks;

	int *space_exceeded = (int*)malloc(sizeof(int));

	T.nseq_offsets = vector<int>(numBlocks);

	// memcpy reference graph
	for (int b = 0; b < numBlocks; b++) {
		size_t offset_letters = b * g->dyn_len_global;
		size_t offset_edges = b * g->edgesNumber;
		size_t offset_bounds = b * (g->dyn_len_global + 1);
		size_t offset_len = b;
	
		cudaErrchk(cudaMemcpy(T.dyn_letters_global_d + offset_letters, g->dyn_letters_global, g->dyn_len_global * sizeof(unsigned char), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(T.dyn_edges_global_d + offset_edges, g->dyn_edges_global, g->edgesNumber * sizeof(Edge), cudaMemcpyHostToDevice));	
		cudaErrchk(cudaMemcpy(T.dyn_edge_bounds_global_d + offset_bounds, g->dyn_edge_bounds_global, (g->dyn_len_global + 1) * sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(T.dyn_len_global_d + offset_len, &g->dyn_len_global, sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(T.dyn_nE_global_d + offset_len, &g->edgesNumber, sizeof(int), cudaMemcpyHostToDevice));
	}

	cudaStreamSynchronize(0);

	// assign to device pointers T pointers
	assign_device_memory<<<1, 1>>>(T.dyn_letters_global_d, T.dyn_edges_global_d, T.dyn_edge_bounds_global_d, 
				       T.moves_global_d, T.diagonals_global_sc_d, T.d_offsets_global_d, T.x_to_ys_d, T.y_to_xs_d, 
					   T.dyn_len_global_d, numBlocks, T.seq_offsets_d, T.sequences_d, T.dyn_nE_global_d, seqLen, T.results_d);

	cudaStreamSynchronize(0);


	int block_offset = 0;
	int BLOCKS = numBlocks;
	
	init_kernel_block_parameters(reads, &T.sequences, T.nseq_offsets, T.seq_offsets, &T.tot_nseq, block_offset, numBlocks, batchSize);
	
	cudaErrchk(cudaMemcpy(T.space_exceeded, space_exceeded, sizeof(int), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(T.sequences_d, T.sequences, T.seq_offsets[T.tot_nseq-1], cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(T.seq_offsets_d, T.seq_offsets.data(), T.tot_nseq * sizeof(int), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(T.nseq_offsets_d, T.nseq_offsets.data(), (unsigned long long)BLOCKS * sizeof(int), cudaMemcpyHostToDevice));
	
	cudaStreamSynchronize(0);

	auto memcpy = NOW;
    int c = duration_cast<microseconds>(memcpy - start).count();
    std::cout << "Memcpy duration: " << c << " microseconds" << std::endl;

	// printGraphStructure<<<1, 1>>>(numBlocks, batchSize);

	// cudaStreamSynchronize(0);

	int shared_size = (nV+1 + seqLen+1 + nV+seqLen+1)*sizeof(int) + 
		nE*sizeof(Edge) + seqLen*sizeof(Edge) + seqLen*sizeof(unsigned char);

	long long global_size = sizeof(int) + (unsigned long long)numBlocks * sizeof(int) + (unsigned long long)numBlocks * sizeof(int) +
					(unsigned long long)seqLen * batchSize * numBlocks + (unsigned long long)numBlocks * batchSize * sizeof(int) + 
					(unsigned long long)numBlocks * sizeof(int) + (unsigned long long)nV * numBlocks + (unsigned long long)nE * numBlocks * sizeof(Edge) +
					(unsigned long long)(nV+1) * numBlocks * sizeof(int) + (unsigned long long)2 * (nV+1) * (seqLen+1) * numBlocks * sizeof(unsigned char) + 
					(unsigned long long)(nV+1)*(seqLen+1) * numBlocks * sizeof(short) + (unsigned long long)(nV + seqLen+1) * numBlocks * sizeof(int) + 
					(unsigned long long)nV * numBlocks * sizeof(int) + (unsigned long long)nV * numBlocks * sizeof(int) + (unsigned long long)numBlocks * sizeof(int);

	checkCudaDeviceLimits(seqLen, shared_size, global_size);

	auto start_alignment = NOW;

	int i_seq_idx = 0;

	for(int j_seq_idx = 0; j_seq_idx < batchSize; j_seq_idx++) {

		if(j_seq_idx == batchSize-1 && lastBatch != 0){
			BLOCKS = lastBatch;
		}

		// cout << endl;
		// cout << "BLOCKS = " << BLOCKS << "   numBlocks = " << numBlocks << "   batchSize = " << batchSize << "   j_seq_idx = " << j_seq_idx << endl;		
				
		compute_d_offsets<<<BLOCKS, 1>>>(i_seq_idx, j_seq_idx, T.nseq_offsets_d);
		
		cudaStreamSynchronize(0); 
		
		init_diagonals<<<BLOCKS, 1>>>(i_seq_idx, j_seq_idx, T.uses_global, T.nseq_offsets_d);

		cudaStreamSynchronize(0);

		sw_align<<<BLOCKS, seqLen+1, shared_size>>>(i_seq_idx, j_seq_idx, T.uses_global, T.nseq_offsets_d);
		
		// sw_align<<<BLOCKS, SL+1>>>(i_seq_idx, j_seq_idx, T.uses_global, T.nseq_offsets_d);

		cudaStreamSynchronize(0);

		cudaErrchk(cudaMemcpy(T.results + j_seq_idx * numBlocks, T.results_d, (unsigned long long)BLOCKS * sizeof(int), cudaMemcpyDeviceToHost));
		
		cudaStreamSynchronize(0);
	}

	auto end_alignment = NOW;
    c = duration_cast<microseconds>(end_alignment - start_alignment).count();
    std::cout << "Alignment duration: " << c << " microseconds" << std::endl;
}


#endif