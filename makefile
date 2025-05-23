# Compiler and flags
NVCC = /usr/local/cuda-12/bin/nvcc
CXXFLAGS = -std=c++11 -O2

# Source files
SRC = main.cpp src/utils.cpp src/poagpu.cu

# Output binary
TARGET = poagpu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) src/poagpu.cu -x cu src/utils.cpp src/gfaToGraph.cpp main.cpp -o $(TARGET) $(CXXFLAGS)

# Clean the build
clean:
	rm -f $(TARGET)