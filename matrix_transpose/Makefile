NVCC=nvcc
NVCFLAGS=-std=c++11 -O3 -arch=sm_30 -Xcompiler="-fopenmp"

all: matrix_transpose

matrix_transpose: matrix_transpose.cu
	$(NVCC) $(NVCFLAGS) matrix_transpose.cu -o matrix_transpose

run: matrix_transpose
	./matrix_transpose

clean:
	rm -f matrix_transpose
