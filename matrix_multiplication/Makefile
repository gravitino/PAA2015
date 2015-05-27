NVCC=nvcc
NVCFLAGS=-std=c++11 -O3 -arch=sm_30 -Xcompiler="-fopenmp"

all: matrix_multiplication

matrix_multiplication: matrix_multiplication.cu
	$(NVCC) $(NVCFLAGS) matrix_multiplication.cu -o matrix_multiplication

run: matrix_multiplication
	./matrix_multiplication

clean:
	rm -f matrix_multiplication
