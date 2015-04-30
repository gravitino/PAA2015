#include<iostream>    // cout, endl
#include<algorithm>   // iota, fill
#include<omp.h>       // benchmark below (mutli-threading with openMP pragmas)

///////////////////////////////////////////////////////////////////////////////
// IGNORE THIS HELPERS (taken from https://github.com/gravitino/cudahelpers)
///////////////////////////////////////////////////////////////////////////////

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

// error makro
#define CUERR {                                                              \
    cudaError_t err;                                                         \
    if ((err = cudaGetLastError()) != cudaSuccess) {                         \
       std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "       \
                 << __FILE__ << ", line " << __LINE__ << std::endl;          \
       exit(1);                                                              \
    }                                                                        \
}

// convenient timers
#define TIMERSTART(label)                                                    \
        cudaEvent_t start##label, stop##label;                               \
        float time##label;                                                   \
        cudaEventCreate(&start##label);                                      \
        cudaEventCreate(&stop##label);                                       \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                     \
        cudaEventRecord(stop##label, 0);                                     \
        cudaEventSynchronize(stop##label);                                   \
        cudaEventElapsedTime(&time##label, start##label, stop##label);       \
        std::cout << "#" << time##label                                      \
                  << " ms (" << #label << ")" << std::endl; 


///////////////////////////////////////////////////////////////////////////////
// STUDENTS PART (feel free to code)
// compile with: nvcc vector_add.cu -std=c++11 -O3 -arch=sm_30 \
//               -Xcompiler="-fopenmp" -o vector_add
///////////////////////////////////////////////////////////////////////////////

// 1 gigabytes of memory for float (device 0 has 12G RAM in total)
#define N (1L<<28)

__global__
void add_kernel(float * A, float * B, float * C, size_t n) {
    int thid = blockDim.x*blockIdx.x+threadIdx.x;

    if (thid < n)
        C[thid] = A[thid]+B[thid];
}


__global__
void sgr_kernel(float * A, float * B, float * C, size_t n) {
    int thid = blockDim.x*blockIdx.x+threadIdx.x;

    for (int index = thid; index < n; index += gridDim.x*blockDim.x)
         C[index] = A[index]+B[index];
}

int main () {
    
    // use the first GPU (0..Tesla K40 12G RAM, 1..Titan 6G RAM)
    cudaSetDevice(0);                                                     CUERR

    // small letters for host, capital letters for device memory
    float *a, *b, *c, *A, *B, *C;

    // allocate host memory
    cudaMallocHost(&a, sizeof(float)*N);                                  CUERR
    cudaMallocHost(&b, sizeof(float)*N);                                  CUERR
    cudaMallocHost(&c, sizeof(float)*N);                                  CUERR
    
    // allocate device memory
    cudaMalloc(&A, sizeof(float)*N);                                      CUERR
    cudaMalloc(&B, sizeof(float)*N);                                      CUERR
    cudaMalloc(&C, sizeof(float)*N);                                      CUERR

    // fill a and b with stuff
    TIMERSTART(fillArraysOnHostSide)
    std::iota(a, a+N, 0);             // (0, 1, 2, 3, ..., N-1)
    std::fill(b, b+N, 1);             // (1, 1, 1, 1, ..., 1)
    TIMERSTOP(fillArraysOnHostSide)

    // measure the time for overall execution on GPU
    TIMERSTART(overallCUDA)

    // copy a and b to A and B from host to device
    TIMERSTART(copyHostToDevice)
    cudaMemcpy(A, a, sizeof(float)*N, cudaMemcpyHostToDevice);            CUERR
    cudaMemcpy(B, b, sizeof(float)*N, cudaMemcpyHostToDevice);            CUERR
    TIMERSTOP(copyHostToDevice)

    // Note, the next line is not needed in practice. However, we overwrite
    // the device vector C to prevent spurious false positives. As an example,
    // if another student writes the correct result to C and the GPU assigns
    // the same address range during your run (this happens quite often) then
    // you might pass the test below even if you process nothing!
    cudaMemset(C, 0, sizeof(float)*N);                                    CUERR

    // invoke the kernel
    TIMERSTART(kernelTime)
    add_kernel<<<SDIV(N, 1024), 1024>>>(A, B, C, N);                      CUERR
    TIMERSTOP(kernelTime)

    // if you are bored try to write the kernel for this grid configuration
    TIMERSTART(staticGridKernelTime)
    sgr_kernel<<<1024, 1024>>>(A, B, C, N);                               CUERR
    TIMERSTOP(staticGridKernelTime)

    // copy C to c from device to host
    TIMERSTART(copyDeviceToHost)
    cudaMemcpy(c, C, sizeof(float)*N, cudaMemcpyDeviceToHost);            CUERR
    TIMERSTOP(copyDeviceToHost)

    // stop overall GPU timer and print result
    TIMERSTOP(overallCUDA)

    ///////////////////////////////////////////////////////////////////////////
    // BENCHMARKS AND CHECKS (you may ignore this, especially the openMP part)
    ///////////////////////////////////////////////////////////////////////////

    // check for correct result computed by CUDA
    for (size_t index = 0; index < N; index++) {
        if (c[index] != a[index]+b[index]) {
            std::cout << "error at position " << index << std::endl;
            break;
        }
    }

    // measure time for vector addition on single-threaded host
    TIMERSTART(overallSingleCore)
    for (size_t index = 0; index < N; index++)
        c[index] = a[index]+b[index];
    TIMERSTOP(overallSingleCore)

    // measure time for vector addition on multi-threaded host
    TIMERSTART(overallMultiCore)
    # pragma omp parallel for
    for (size_t index = 0; index < N; index++)
        c[index] = a[index]+b[index];
    TIMERSTOP(overallMultiCore)

    // get rid of the memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    // print status
    float usedMem = 3.0*N*sizeof(float)/(1L<<30);
    std::cout << "#processed " << usedMem << " gigabytes." << std::endl;
    std::cout << "CUDA programming is fun!" << std::endl;
}
