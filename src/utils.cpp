#include <unordered_map>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <map>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "../include/poagpu.cuh"

int check_input_int(string &arg){
	
	try{
		size_t pos;
		int arg_i = stoi(arg, &pos);
		if(pos < arg.size()){
			std::cerr << "Trailing characters after number: " << arg << '\n';
		}
		return arg_i;
	} catch (invalid_argument const &ex) {
		std::cerr << "Invalid number: " << arg << '\n';
		return -1;
	} catch (out_of_range const &ex) {
		std::cerr << "Number out of range: " << arg << '\n';
		return -1;
	}
	
}

void print_reads(vector<vector<string>> reads) {

	cout << "Printing reads: "<< endl;
	for(int i = 0; i < reads.size(); i++) {
		cout << i << endl;
		cout << "Batch " << i << endl;
		for(int j = 0; j < reads[i].size(); j++) {
			cout << reads[i][j] << endl;
		}
	}
}


void read_batch(vector<vector<string>> &reads, size_t size, string filename){

	ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Errore: Impossibile aprire il file " << filename << endl;
        return;
    }

    string line;
    vector<string> allReads;

    while (getline(infile, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        if (!line.empty())
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
        if (line.empty() || line[0] == '>')
            continue;
        allReads.push_back(line);
    }
    infile.close();

    int total = allReads.size();
    int base = total / size;
    int remainder = total % size;

    int index = 0;
    for (size_t group = 0; group < size; group++) {
        int groupSize = base + (group < remainder ? 1 : 0);
        vector<string> groupReads;
        for (int j = 0; j < groupSize; j++) {
            groupReads.push_back(allReads[index++]);
        }
        reads.push_back(groupReads);
    }
}

void get_bmean_batch_result_gpu(vector<vector<string>> reads, int &c, const int numBlocks, int batchSize, graph_h* g){

	poa_gpu_utils::TaskRefs T;

	auto start = NOW;

    size_t seqLen = 0;
    for (const auto& batch : reads) {
        for (const auto& read : batch) {
            if (read.size() > seqLen)
                seqLen = read.size();
        }
    }

	gpu_POA_alloc(T, numBlocks, batchSize, g->dyn_len_global, g->edgesNumber, seqLen);
    auto alloc = NOW;
    c = duration_cast<microseconds>(alloc - start).count();
	std::cout << "Alloc duration: " << c << " microseconds" << std::endl;

	gpu_POA(reads, T, numBlocks, batchSize, g, seqLen);

	gpu_POA_free(T);

	auto end = NOW;
	c = duration_cast<microseconds>(end - start).count();
	std::cout << "Total duration: " << c << " microseconds" << std::endl;
}