/**
 * @file gfaToGraph.cpp
 * @brief This file contains the implementation of a program that reads a GFA (Graphical Fragment Assembly) file and converts it into a graph representation.
 * 
 * The program reads the GFA file, counts the number of vertices and edges, and stores the sequences and links between nodes. It then constructs an adjacency list for the graph and prints the adjacency list, vertex labels, in-neighbors, and in-offsets.
 * 
 * The program expects a single command-line argument which is the path to the GFA file.
 * 
 * Usage:
 * @code
 * ./gfaToGraph <GFA file>
 * @endcode
 * 
 * The program performs the following steps:
 * 1. Reads the GFA file and counts the number of vertices and edges.
 * 2. Stores the sequences of the nodes.
 * 3. Constructs an adjacency list for the graph.
 * 4. Sorts the adjacency list using qsort.
 * 5. Copies the adjacency list to an in-neighbors array.
 * 6. Prints the adjacency list, vertex labels, in-neighbors, and in-offsets.
 * 
 * The program uses the following data structures:
 * - `unordered_map<std::string, int> nodeIdToIndex`: A hash map to store the link between the node ID and the sequence of the node.
 * - `map<int, seqInfo> seqs`: A map to store the start and end of the sequence for each node.
 * - `vector<int>* adjacencyList`: A temporary adjacency list to store the edges for every vertex.
 * - `char* vertexLabels`: An array to store the labels of the vertices.
 * - `int* inNeigh`: An array to store the in-neighbors of the vertices.
 * - `int* inOffset`: An array to store the offsets of the in-neighbors.
 * 
 * The program also defines a `seqInfo` struct to store the start and end of the sequence for each node.
 * 
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return Returns 0 on success, or 1 on failure.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include "../include/gfaToGraph.h"

using namespace std;

// struct to store the start and end of the sequence
typedef struct{
	// int idx;
	int start;
	int end;
}seqInfo;


void convertGFAtoGraph(graph_h* g, const string& filename){

    // if (argc != 2) {
    //     cerr << "Usage: " << argv[0] << " <GFA file>" << endl;
    //     return 1;
    // }

    // string filename = argv[1];
    ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        return;
    }

	// to count the number of vertices and edges
    int vertexNumber = 0, edgesNumber = 0;	

    string line;

	// hash map to store the link between the node id and the sequence of the node
	unordered_map<std::string, int> nodeIdToIndex; // hash map `nodeId` -> progressive index

	// map to store a specifc struct to store the start and end of the sequence
	// the key is the index of the node from the hash map above
	map<int, seqInfo> seqs;
	
	string vertexLab;

    // first read to count the number of vertices and edges and store the sequences
    while(getline(file, line)){

		// Skip blank or whitespace-only lines
		if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
		istringstream iss(line);

        char type;
		// iss >> type;

		// Attempt to read a non-whitespace character; skip if none
		if (!(iss >> type)) continue;

        if(type == 'S'){
            std::string nodeId, sequence;
            iss >> nodeId >> sequence;
			
			// probably not necessary because nodeId is unique
            if (nodeIdToIndex.find(nodeId) == nodeIdToIndex.end()) {
                nodeIdToIndex[nodeId] = vertexNumber; // assign the index to the node

				for(char c : sequence){
					vertexLab.push_back(c);
				}
				seqInfo s;
				
				s.start=vertexNumber;
				vertexNumber += sequence.size();
				s.end=vertexNumber - 1;
				seqs.insert({nodeIdToIndex[nodeId], s});
			}
        }
    }

	// temporary adjacency list to store the edges for evrey vertex
	vector<int>* adjacencyList = new vector<int>[vertexNumber];

	// second read to store the edges

	file.clear();
	file.seekg(0, ios::beg);

	adjacencyList[0].push_back(-1);
	edgesNumber++;

	while(getline(file, line)){

		// Skip blank or whitespace-only lines. Prefer to put it here in order to avoid skipping of blank lines in the middle of the file
		if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
		istringstream iss(line);
		char type;
		// iss >> type;

		// Attempt to read a non-whitespace character; skip if none
		if (!(iss >> type)) continue;

		// inserting the edges to the adjacency list for the single vertex in the sequence
		if (type == 'S') {
			string nodeId;
			iss >> nodeId;

			int start = seqs[nodeIdToIndex[nodeId]].start;
			int end = seqs[nodeIdToIndex[nodeId]].end;

			for (int i = end; i > start; i--) {
				adjacencyList[i].push_back(i - 1);
				edgesNumber++;
			}
    	}
		else if (type == 'L'){
			// Link: L <from_node> <from_orient> <to_node> <to_orient> <overlap>
			string fromNode, toNode;
            char fromOrient, toOrient;
            // string overlap; // not used

            iss >> fromNode >> fromOrient >> toNode >> toOrient;

            int fromIndex = nodeIdToIndex[fromNode];
            int toIndex = nodeIdToIndex[toNode];

			adjacencyList[seqs[toIndex].start].push_back(seqs[fromIndex].end);
			edgesNumber++;
		}
	}

    file.close();

	// char* vertexLabels = new char[vertexNumber];
	// int* inNeigh = new int[edgesNumber];
	// int* inOffset = new int[vertexNumber + 1];

	g->dyn_edge_bounds_global = new int[vertexNumber + 1];
	g->dyn_edges_global = new Edge[edgesNumber]; 
	g->dyn_letters_global = new unsigned char[vertexNumber];
	
	g->dyn_len_global = vertexNumber;
	g->edgesNumber = edgesNumber;

	// calculate inOffset array
	// inOffset[0] = 0;	// first one is always 0
	g->dyn_edge_bounds_global[0] = 0;
	for (int i = 1; i < vertexNumber + 1; i++) {
        // inOffset[i] = inOffset[i - 1] + edgeCounts[i - 1];
		// inOffset[i] = inOffset[i - 1] + adjacencyList[i - 1].size();
		g->dyn_edge_bounds_global[i] = g->dyn_edge_bounds_global[i - 1] + adjacencyList[i - 1].size();
    }

	// sort the links in the adjacency list for every vertex using qsort
	for(int i=0; i<vertexNumber; i++){
		// sort(adjacencyList[i].begin(), adjacencyList[i].end());

		// Convert vector to array for qsort
		int* tempArray = new int[adjacencyList[i].size()];
		copy(adjacencyList[i].begin(), adjacencyList[i].end(), tempArray);

		qsort(tempArray, adjacencyList[i].size(), sizeof(int), [](const void* a, const void* b) {
			return (*(int*)a - *(int*)b);
		});

		// Copy sorted array back to vector
		copy(tempArray, tempArray + adjacencyList[i].size(), adjacencyList[i].begin());
		delete[] tempArray;
	}

	// copy the adjacency list to inNeigh array
	for(int i = 0, currentVertex = 0; i < vertexNumber; i++){
		for(int neighbor : adjacencyList[i]){
			// inNeigh[currentVertex] = neighbor;
			// Warning: Conversione implicita da int a short, che potrebbe causare perdita di dati se neighbor > 32767.
			g->dyn_edges_global[currentVertex] = Edge{(short)neighbor}; // Assuming Edge is defined as a struct
			currentVertex++;
		}
	}

	// copy vertex labels
	memcpy(g->dyn_letters_global, vertexLab.c_str(), vertexLab.size());

	// initialize edge_bounds and end_nodes
	// for (int i = 0; i < vertexNumber; i++) {
	// 	g->edge_bounds[i] = adjacencyList[i].size();
	// 	g->end_nodes[i] = adjacencyList[i].empty() ? 1 : 0; // 1 if no outgoing edges
	// 	g->sequence_ids[i] = i; // Assuming sequence IDs are just the indices
	// }
	

	// print the adjacency list
	// for (int i = 0; i < vertexNumber; i++) {
	// 	cout << "Adjacency list of vertex " << i << ": ";
	// 	for (int neighbor : adjacencyList[i]) {
	// 		cout << neighbor << " ";
	// 	}
	// 	cout << endl;
	// }

	// cout << "Vertex Number: " << vertexNumber << endl;
	// cout << "Edges Number: " << edgesNumber << endl;

	// strcpy(vertexLabels, vertexLab.c_str());
	
	// cout << "Vertex Labels: ";

	// for(int i = 0; i < vertexNumber; i++){
	// 	cout << vertexLabels[i] << (i != vertexNumber - 1 ? ", " : "\n");
	// }

	// cout <<"In Neigh: ";
	// for(int i = 0; i < edgesNumber; i++){
	// 	cout << inNeigh[i] << (i != edgesNumber - 1 ? ", " : "\n");
	// }

	// cout << "In Offset: ";
	// for(int i = 0; i < vertexNumber + 1; i++){
	// 	cout << inOffset[i] << (i != vertexNumber ? ", " : "\n");
	// }

	// cout << "Graph structure: " << endl;

	// for(int i = 0; i < vertexNumber; i++){
	// 	cout << g->dyn_letters_global[i] << (i != vertexNumber - 1 ? ", " : "\n");
	// }

	// cout <<"In Neigh: ";
	// for(int i = 0; i < edgesNumber; i++){
	// 	cout << g->dyn_edges_global[i] << (i != edgesNumber - 1 ? ", " : "\n");
	// }

	// cout << "In Offset: ";
	// for(int i = 0; i < vertexNumber + 1; i++){
	// 	cout << g->dyn_edge_bounds_global[i] << (i != vertexNumber ? ", " : "\n");
	// }

	// delete[] vertexLabels;
	// delete[] inNeigh;
	// delete[] inOffset;
	delete[] adjacencyList;

    return;
}