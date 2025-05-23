# ALIGNER-INTRA-GPU

GPU-accelerated sequence-to-graph aligner based on partial order dynamic programming. This project implements exact alignment (no heuristics) between reads and reference graphs using CUDA, efficiently parallelizing both intra- and inter-sequence computations.

---

## 🧬 Overview

This tool computes the optimal alignment score between a sequence and a genomic variation graph, represented in CSR format. It guarantees exact results by filling the full dynamic programming matrix. The implementation is optimized for NVIDIA GPUs and scales well across both short and long reads.

---

## 🔧 Build Instructions

To build the project:

```bash
git clone https://github.com/necst/GPU-POA.git
cd ALIGNER-INTRA-GPU
make
```

To clean the build:
```bash
make clean
```

## 🚀 Usage

### 🧪 Manual test with custom inputs

You can run the aligner directly on any input files using:

```bash
./poagpu <num_blocks> <reads_file.fa> <graph_file.gfa>
```

Arguments:
- `<graph_file.gfa>`: input graph in GFA format
- `<reads_file.fa>`: input reads in FASTA format
- `<num_blocks>`: number of CUDA blocks to use

This command allows you to test the aligner with any graph and read set of your choice.

### ⚙️ Automatic test using our datasets
Alternatively, you can test the aligner using our predefined synthetic datasets with the following Python script:

```bash
python scripts/run_poagpu.py <num_blocks> <num_vertici> <num_reads> <len_reads> [--example]
```
Arguments:

- `<num_blocks>`: number of CUDA blocks to use

- `<num_vertici>`: number of vertices in the graph (as indicated in the filename)

- `<num_reads>`: number of reads (e.g. 100, 1000, 10000 — will be auto-formatted to the find the right folder)

- `<len_reads>`: read length (as indicated in the filename)

- `--example` (optional): use example files from test/examples/ directory

This script allows you to reproduce our benchmarks or run structured tests with minimal manual setup.


## 📁 Project Structure

- `src/` — CUDA and C++ source files  
- `include/` — Header files  
- `test/` — Example input graphs and read sets    

---

## 👨‍💻 Authors

Developed by **Leonardo Tisato**  and **Gabriele Amodeo**.

Supervised by **Ph.D Student Mirko Coggi** and **Prof. Marco Domenico Santambrogio**.

---



