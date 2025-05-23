#include "../include/poa.h"
// #include "assert.h"
#include <cuda_runtime.h>
#include <iostream>


#define MATCH 1
#define MISMATCH -2
#define GAP 1

#define USES_GLOBAL 1

#define wrapSize 32
#define FULL_MASK 0xffffffff

struct MaxCell {
	int val;
	int x;
	int y;
};

static const int NOT_ALIGNED = -1;

__device__ int score(unsigned char i, unsigned char j) { return i == j ? MATCH : MISMATCH; }

__device__ char* reads_d;
__device__ int* read_offsets;

__device__ int seqLen;

__device__ int* dyn_len_global;
__device__ int* dyn_nE_global;
__device__ char* dyn_letters_global;
__device__ Edge* dyn_edges_global;
__device__ int* dyn_edge_bounds_global;

__device__ unsigned char* moves_x_global;
__device__ unsigned char* moves_y_global;
__device__ short* diagonals_sc_global;

__device__ int* d_offsets_global;
__device__ int* x_to_ys;
__device__ int* y_to_xs;
__device__ int g_space_exceeded = 0;

__device__ int* result;


__global__ void printGraphStructure(int numBlocks, int batchSize) {
	int myId = blockIdx.x;
	int nV = dyn_len_global[myId];
	int nE = dyn_nE_global[myId];

	printf("\n--- seqLen: %d ---\n", seqLen);

	printf("\nPrinting graph: \n");
	printf("nV: ");
	for(int i = 0; i < numBlocks; i++) {
		printf("%d ", dyn_len_global[i]);
	}
	printf("\n");
	printf("nE: ");
	for(int i = 0; i < numBlocks; i++) {
		printf("%d ", dyn_nE_global[i]);
	}
	printf("\n");
	printf("Letters: ");
	for(int i = 0; i < nV * numBlocks; i++) {
		printf("%c ", dyn_letters_global[i]);
	}
	printf("\n");
	printf("Edge: ");
	for(int i = 0; i < nE * numBlocks; i++) {
		printf("%hd ", dyn_edges_global[i]);
	}
	printf("\n");
	printf("Edge Bounds: ");
	for(int i = 0; i < (nV + 1) * numBlocks; i++) {
		printf("%d ", dyn_edge_bounds_global[i]);
	}
	printf("\n");
}


__global__ void assign_device_memory( char* dletters, Edge* dedges, int* dedgebounds, unsigned char* moves, short* diagonals_sc, 
									int* d_offs, int* xy, int* yx, int* dynlg, const int num_blocks, int* roffs, char* reads, int* dynne, int seqlen, int* res){
	
	int offset = 0;
	for(int i = 0; i < num_blocks; i++) {
		offset += dynlg[i];
	}

	reads_d = reads;
	read_offsets = roffs;
	
	dyn_letters_global = dletters;
	dyn_edges_global = dedges;
	dyn_edge_bounds_global = dedgebounds;

	seqLen = seqlen;

	moves_x_global = moves;
	moves_y_global = moves + (unsigned long)(offset + num_blocks) * (seqLen + 1); 
	diagonals_sc_global = diagonals_sc;
	
	d_offsets_global = d_offs;
	x_to_ys = xy;
	y_to_xs = yx;

	dyn_len_global = dynlg;
	dyn_nE_global = dynne;

	result = res;

}


__global__ void init_diagonals(int i_seq_idx, int j_seq_idx, int uses_global, int* nseq_offsets){
	
	if(g_space_exceeded) return;

	__shared__ int local_seqLen;
	if (threadIdx.x == 0)
		local_seqLen = seqLen;
	__syncthreads();

	int nseq;
	int block_offset;
	int myId = blockIdx.x;
	int len_y;
	int len_x;

	int nV = dyn_len_global[myId];
	int nE = dyn_nE_global[myId];

	
	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = dyn_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}

	if(j_seq_idx < nseq){
	
		len_y = read_offsets[block_offset + j_seq_idx] - read_offsets[block_offset + j_seq_idx-1];
	
		Edge* left_x = dyn_edges_global + nE * myId; 
		int* lx_start = dyn_edge_bounds_global + (nV+1) * myId;

		short* diagonals_sc = diagonals_sc_global + (nV+1)*(local_seqLen+1) * myId;
		int* d_offsets = d_offsets_global + (nV+local_seqLen+1) * myId;
	
		int min_d = len_x < len_y ? len_x : len_y;

		diagonals_sc[0] = 0;
		
		int min_score = -999999;
		int try_score;   

		for (int i = 1; i < len_x + 1; i++) {

			int offs = i < min_d ? i : min_d;
			short &curr_cell_sc = (diagonals_sc+d_offsets[i])[offs];
		
			curr_cell_sc = min_score;

			int k = lx_start[i - 1];
			for (int x_count = 1; k < lx_start[i]; k++, x_count++) {

				Edge xl = left_x[k];
				int prev_last_cell = xl + 1 < min_d ? xl + 1 : min_d;
				short prev_sc = (diagonals_sc + d_offsets[xl + 1])[prev_last_cell];
			
				try_score = prev_sc - GAP;    

				if (try_score > curr_cell_sc) {
					curr_cell_sc = try_score;
				}
			}
		}

		for (int i = 1; i < len_y + 1; i++) {

			short &curr_cell_sc = diagonals_sc[d_offsets[i]];
			curr_cell_sc = min_score;

			int prev = i - 2;
			short prev_sc = diagonals_sc[d_offsets[prev + 1]];
			try_score = prev_sc - GAP;
			if (try_score > curr_cell_sc) {
				curr_cell_sc = try_score;
			}
		}
	}
}

__device__ void trace_back_lpo_alignment(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left,
					 int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets) {

	int xmove, ymove;
	Edge left;
	
	for (int i = 0; i < len_x; i++) {
		x_to_y[i] = (int)NOT_ALIGNED;
	}

	for (int i = 0; i < len_y; i++) {
		y_to_x[i] = (int)NOT_ALIGNED;
	}

	while (best_x >= 0 && best_y >= 0) {
		
		int diagonal = best_x + best_y + 2;
		int offset = (diagonal <= len_y) ? best_x+1 : best_x+1 - (diagonal - len_y); 
		xmove = (move_x+d_offsets[diagonal])[offset];
		ymove = (move_y+d_offsets[diagonal])[offset];

		if (xmove > 0 && ymove > 0) { /* ALIGNED! MAP best_x <--> best_y */
			x_to_y[best_x] = best_y;
			y_to_x[best_y] = best_x;
		}

		if (xmove == 0 && ymove == 0) { /* FIRST ALIGNED PAIR */
			x_to_y[best_x] = best_y;
			y_to_x[best_y] = best_x;
			break;  /* FOUND START OF ALIGNED REGION, SO WE'RE DONE */
		}

		if (xmove > 0) { /* TRACE BACK ON X */
			int start = start_x[best_x];
			while ((--xmove) > 0) {
				start++;
			}
			left = x_left[start];
			best_x = left;
		}

		if (ymove > 0) { /* TRACE BACK ON Y */
			int start = start_y[best_y];
			while ((--ymove) > 0) {
				start++;
			}
			left = y_left[start];
			best_y = left;
		}
	}
	return;
}

__inline__ __device__ MaxCell wrapReduceMax(MaxCell cell){
	
	for(int offset = wrapSize / 2; offset > 0; offset /= 2){
		short val = __shfl_down_sync(FULL_MASK, cell.val, offset);
		int x = __shfl_down_sync(FULL_MASK, cell.x, offset);
		int y = __shfl_down_sync(FULL_MASK, cell.y, offset);
		int is_max = val >= cell.val && ( val > cell.val | x < cell.x | y < cell.y );
		cell.val = is_max ? val : cell.val;
		cell.x = is_max ? x : cell.x;
		cell.y = is_max ? y : cell.y;
	}
	return cell;
}

template<int N_WRAPS>
__inline__ __device__ MaxCell blockReduceMax(MaxCell cell){
	
	static __shared__ MaxCell shared_max[N_WRAPS];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	
	cell = wrapReduceMax(cell);
	
	if(lane == 0){
		shared_max[wid] = cell;	
	}
	__syncthreads();

	MaxCell zero_cell = { -999999, -1, -1 };
	cell = (threadIdx.x < blockDim.x / warpSize) ? shared_max[lane] : zero_cell;
	
	if(wid == 0) {
		cell = wrapReduceMax(cell);
	}
	return cell;
}

__inline__ __device__ MaxCell blockReduceMax_dynamic(MaxCell cell, int num_wraps) {
    extern __shared__ MaxCell shared_max[];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    cell = wrapReduceMax(cell);
    if (lane == 0) {
        shared_max[wid] = cell;
    }
    __syncthreads();

    MaxCell zero_cell = { -999999, -1, -1 };
    cell = (threadIdx.x < blockDim.x / warpSize) ? shared_max[lane] : zero_cell;

    if (wid == 0) {
        cell = wrapReduceMax(cell);
    }
    return cell;
}

 __global__ void sw_align(int i_seq_idx, int j_seq_idx, int uses_global, int* nseq_offsets) {
	
	if(g_space_exceeded) return;

	int myId = blockIdx.x;

	int nV = dyn_len_global[myId];
	int nE = dyn_nE_global[myId];


	__shared__ int nseq;
	__shared__ int block_offset;
	__shared__ int y_seq_offs;
	__shared__ int len_y;
	__shared__ int len_x;

	__shared__ int local_seqLen;


	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = dyn_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}
	
	
	if(j_seq_idx < nseq){
		
		y_seq_offs = read_offsets[block_offset + j_seq_idx-1];
		len_y = read_offsets[block_offset + j_seq_idx] - read_offsets[block_offset + j_seq_idx-1];
		
		int c = threadIdx.x;

		__shared__ char* seq_x;
		__shared__ int* x_to_y;
		__shared__ int* y_to_x;
		__shared__ short* diagonals_sc;
		__shared__ unsigned char* moves_x;
		__shared__ unsigned char* moves_y;

		if(c==0){
			local_seqLen = seqLen;
			seq_x = dyn_letters_global + nV * myId;
			x_to_y= x_to_ys + nV * myId;
			y_to_x = y_to_xs + nV * myId;
			diagonals_sc = diagonals_sc_global + (nV+1)*(local_seqLen+1) * myId;
			moves_x = moves_x_global + (nV+1)*(local_seqLen+1) * myId;
			moves_y = moves_y_global + (nV+1)*(local_seqLen+1) * myId;
		}
		__syncthreads();

		extern __shared__ unsigned char shared_mem[]; // Raw shared memory

		int* lx_start = (int*)shared_mem;
		int* ly_start = (int*)&lx_start[nV + 1];
		int* d_offsets = (int*)&ly_start[local_seqLen + 1];
		Edge* left_x = (Edge*)&d_offsets[nV + local_seqLen + 1];
		Edge* left_y = (Edge*)&left_x[nE];
		unsigned char* seq_y = (unsigned char*)&left_y[local_seqLen];
		
		int offs = 0;
		do{
			if(offs + c < len_x+1){
				lx_start[offs + c] = (dyn_edge_bounds_global + (nV+1) * myId)[offs+c];
			}	
			offs += local_seqLen+1;
		}while(offs < len_x+1);

		if(c < len_y+1){
			ly_start[c] = c;
		}
		__syncthreads();
		int x_left_dim = lx_start[len_x];
		int y_left_dim = ly_start[len_y];

		offs = 0;
		do{
			if(offs + c < x_left_dim){
				left_x[offs+c] = (dyn_edges_global + nE * myId)[offs+c];
			}
			offs +=local_seqLen+1;
		}while(offs < x_left_dim);

		offs = 0;
		do{
			if(offs + c < y_left_dim){
				left_y[offs+c] = c - 1;
			}
			offs +=local_seqLen+1;
		}while(offs < y_left_dim);
		
		if(c < len_y){
			seq_y[c] = (reads_d + y_seq_offs)[c];
		}

		offs = 0;
		do{
			if(offs + c < len_x+len_y+1){
				d_offsets[c+offs] = (d_offsets_global + (nV+local_seqLen+1) * myId)[c+offs];
			}
			offs += local_seqLen+1;
		}while(offs < len_x+len_y+1);
		
		MaxCell maxc = { -999999, -1, -1 };
		int min_d = len_x < len_y ? len_x : len_y;

		__syncthreads();

		for (int n = 2; n < len_x + len_y + 1; n++) {

			int lower_bound = n <= len_y;
			int upper_bound = min_d+lower_bound < n ? min_d+lower_bound : n;
			upper_bound = upper_bound < len_x + len_y + 1 - n ? upper_bound : len_x + len_y + 1 - n;

			if (c >= lower_bound && c < upper_bound) {

				int match_score = ((uses_global == 0)-1) & (-999999);  // matchScore = -999999
				
				int match_x = 0;
				int match_y = 0;

				int insert_x_score = -999999;
				int insert_y_score = -999999;
				int insert_x_x = 0;
				int insert_y_y = 0;

				int try_score = -999999;
				
				int j = c + (((n - len_y < 0)-1) & (n - len_y));
				int i = n - j;

				int k = ly_start[i-1];
				int y_count = 1;

				int i_prev = left_y[k] + 1;
				k = ((0 > i_prev + j - len_y)-1) & (i_prev + j - len_y);
				int n_prev = i_prev + j;
				int c_prev = j - k;

				try_score = (diagonals_sc + d_offsets[n_prev])[c_prev];
				try_score -= GAP;
				
				if (try_score > insert_y_score) {
					insert_y_score = try_score;
					insert_y_y = y_count;
				}

				k = lx_start[j - 1];
				for (int x_count = 1; k < lx_start[j]; k++, x_count++) {

					Edge xl = left_x[k];
					int j_prev = xl + 1;
					int k = ((0 > j_prev + i - len_y)-1) & (j_prev + i - len_y);
					int n_prev = j_prev + i;
					int c_prev = j_prev - k;

					try_score = (diagonals_sc + d_offsets[n_prev])[c_prev];
					
					try_score -= GAP;
					
					if (try_score > insert_x_score) {
						insert_x_score = try_score;
						insert_x_x = x_count;
					}

					k = ly_start[i-1];

					int i_prev = left_y[k] + 1;

					k = ((0 > i_prev + j_prev - len_y)-1) & (i_prev + j_prev - len_y);
					n_prev = i_prev + j_prev;
					c_prev = j_prev - k;

					try_score = (diagonals_sc + d_offsets[n_prev])[c_prev];	

					if (try_score > match_score) {
						match_score = try_score;
						match_x = x_count;
						match_y = y_count;
					}
				}

				match_score += score(seq_x[j - 1], seq_y[i - 1]); 
				
				unsigned char my_move_x; 
				unsigned char my_move_y; 
				
				short my_score;
				
				int match_mask = (match_score <= insert_y_score || match_score <= insert_x_score)-1;
				int ins_x_mask = (insert_x_score < match_score || insert_x_score <= insert_y_score)-1;
				int ins_y_mask = (insert_y_score < match_score || insert_y_score < insert_x_score)-1;

				my_score = (match_score & match_mask) + (insert_x_score & ins_x_mask) + (insert_y_score & ins_y_mask);
				
				my_move_x = (match_x & match_mask) + (insert_x_x & ins_x_mask) + 0;
				
				my_move_y = (match_y & match_mask) + (insert_y_y & ins_y_mask) + 0;
				
				if (my_score >= maxc.val) {
					if (my_score > maxc.val ||
						(j-1 == maxc.x && i-1 < maxc.y) || j-1 < maxc.x) {
						maxc.val = my_score;
						maxc.x = j-1;
						maxc.y = i-1;
					}
				}
				(moves_x+d_offsets[n])[c] = my_move_x;
				(moves_y+d_offsets[n])[c] = my_move_y;
				(diagonals_sc + d_offsets[n])[c] = my_score;
			}
			__syncthreads();

			// Printing DP-matrix
			// if (threadIdx.x == 0 && blockIdx.x == 0) {
			// 	int end = (n<(len_x + len_y)) ? d_offsets[n + 1] : (len_x+1) * (len_y+1);
			// 	printf("Diagonal %d:", n);
			// 	for (int d = d_offsets[n]; d < end; d++) {
			// 		printf("(%d)", diagonals_sc[d]);
			// 	}
			// 	printf("\n");
			// }
			// __syncthreads();
		}

		if(c==0) {
			result[myId] = maxc.val;
		}
		
		// max = blockReduceMax<(((SL-1) / wrapSize) + 1)>(max);

		int wraps = ((seqLen - 1) / 32) + 1;
		maxc = blockReduceMax_dynamic(maxc, wraps);
		
		if(c==0){
			trace_back_lpo_alignment(len_x, len_y, moves_x, moves_y, left_x, left_y, lx_start, ly_start, 
									maxc.x, maxc.y, x_to_y, y_to_x, d_offsets);
		}

	} //thread execution if-end
	
	return;
}


__global__ void compute_d_offsets(int i_seq_idx, int j_seq_idx, int* nseq_offsets) {
	
	if(g_space_exceeded) return;

	int nseq;
	int block_offset;
	int myId = blockIdx.x;
	int len_x;
	int len_y;

	__shared__ int local_seqLen;
	local_seqLen = seqLen;
	
	int nV = dyn_len_global[myId];

	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = dyn_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}
	if(j_seq_idx < nseq){
		
		len_y = read_offsets[block_offset + j_seq_idx] - read_offsets[block_offset + j_seq_idx-1];

		int* d_offsets = d_offsets_global + (nV + local_seqLen+1) * myId;  

		int min_d = len_x < len_y ? len_x : len_y;
		int offset = 0;
		for (int j = 0; j < len_x + len_y + 1; j++) {
			int n = min_d + 1 < j + 1 ? min_d + 1 : j + 1;
			n = n < len_x + len_y + 1 - j ? n : len_x + len_y + 1 - j;
			d_offsets[j] = offset;
			offset += n;
		}
	}
}