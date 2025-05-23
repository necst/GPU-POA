#ifndef TASK_H
#define TASK_H
#include<map>
#include<vector>

#define USES_GLOBAL 1

#define MATCH 1
#define MISMATCH -2
#define GAP 1

using namespace std;


namespace poa_gpu_utils{

typedef short Edge; 

struct TaskRefs{
	
	int uses_global = USES_GLOBAL;
	
	vector<int> nseq_offsets;
	int tot_nseq = 0;
	char* sequences;
	vector<int> seq_offsets;

	int* space_exceeded;
	
	int* results;
	
	int* nseq_offsets_d;
	char* sequences_d;
	int* seq_offsets_d;
	
	int* results_d;

	char* dyn_letters_global_d;	
	Edge* dyn_edges_global_d;
	int* dyn_edge_bounds_global_d;

	unsigned char* moves_global_d;	
	short* diagonals_global_sc_d;

	int* d_offsets_global_d;
	int* x_to_ys_d;
	int* y_to_xs_d;
	
	int* dyn_len_global_d;
	int* dyn_nE_global_d;

	int seqLen;
};

} //end poa_gpu_utils

#endif
	
