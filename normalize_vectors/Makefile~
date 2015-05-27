NVCC=nvcc
NVCFLAGS=-std=c++11 -O3 -arch=sm_30 -Xcompiler="-fopenmp"

all: vector_add

vector_add: vector_add.cu
	$(NVCC) $(NVCFLAGS) vector_add.cu -o vector_add

run: vector_add
	./vector_add

clean:
	rm -f vector_add
