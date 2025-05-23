#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <map>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <numeric>
#include <stdexcept>
#include "include/poa.h"
#include "include/gfaToGraph.h"



using namespace std;
using namespace chrono;

#define NOW high_resolution_clock::now()


int main(int argc, char* argv[]) {

	char* path_ref;
	char* path_graph;
	
	if(argc < 4){
		cout << "Invalid arguments. Call this program as: ./poagpu numBlocks read_file.fa graph_file.gfa" << endl;
		return 0;
	}
	
	string max_w_size = argv[1];
	const int NUM_BLOCKS = check_input_int(max_w_size);		

	if(NUM_BLOCKS < 0){
		cout << "Invalid numBlocks size provided" << endl;
		return 0;
	}

	path_ref = argv[2];
	path_graph = argv[3];

	vector<vector<string>> reads;
	
	string filepath_r(path_ref);
	read_batch(reads, NUM_BLOCKS, filepath_r);
	
	// print_reads(reads);

	int numReads = 0;
	for(int i = 0; i < reads.size(); i++) {
		numReads += reads[i].size();
	}

	graph_h g;
	string filepath_g(path_graph);
	auto start = NOW;
	convertGFAtoGraph(&g, filepath_g);
	auto convert = NOW;

    double t = duration_cast<microseconds>(convert - start).count();
    std::cout << "Conversion duration: " << t << " microseconds" << std::endl;

	int batchSize = (numReads - 1) / NUM_BLOCKS + 1;

	int c = 0;

	get_bmean_batch_result_gpu(reads, c, NUM_BLOCKS, batchSize, &g);

	delete[] g.dyn_letters_global;
	delete[] g.dyn_edges_global;
	delete[] g.dyn_edge_bounds_global;
	
	return 0;
}