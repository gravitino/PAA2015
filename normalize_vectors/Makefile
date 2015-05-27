NVCC=nvcc
NVCFLAGS=-std=c++11 -O3 -arch=sm_30 -Xcompiler="-fopenmp"

all: normalize_vectors

normalize_vectors: normalize_vectors.cu
	$(NVCC) $(NVCFLAGS) normalize_vectors.cu -o normalize_vectors

run: normalize_vectors
	./normalize_vectors

clean:
	rm -f normalize_vectors
