#ifndef GFA_TO_GRAPH_H
#define GFA_TO_GRAPH_H

#include <string>  // Se la funzione usa std::string

using namespace std;

typedef short Edge; 

struct graph{
	int* dyn_edge_bounds_global;
	Edge* dyn_edges_global;
	unsigned char* dyn_letters_global;
	int dyn_len_global; // vertices number
	int edgesNumber;
} typedef graph_h;

// Dichiarazione della funzione (ex-main)
void convertGFAtoGraph(graph_h* g, const string& filename);

// void printGraph();

#endif // GFA_TO_GRAPH_H