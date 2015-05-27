#include<iostream>    // cout, endl
#include<algorithm>   // iota
#include<cmath>       // sqrt
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
// compile with: nvcc matrix_transpose.cu -std=c++11 -O3 -arch=sm_30 \
//               -Xcompiler="-fopenmp" -o matrix_transpose
///////////////////////////////////////////////////////////////////////////////

// 4 gigabytes of memory for float (device 0 has 12G RAM in total)
#define N (3*(1L<<13))

// tile size
#define TILE (4)

__global__
void transpose_kernel(float * A, float * B, size_t n) {

    const size_t thid_x = blockDim.x*blockIdx.x+threadIdx.x,
                 thid_y = blockDim.y*blockIdx.y+threadIdx.y;

    if (thid_x < n && thid_y < n)
        B[thid_y*n+thid_x] = A[thid_x*n+thid_y];
}


__global__
void transpose_kernel_tiled(float * A, float * B, size_t m) {

    const size_t thid_x = blockDim.x*blockIdx.x+threadIdx.x,
                 thid_y = blockDim.y*blockIdx.y+threadIdx.y;

    if (thid_x < m && thid_y < m)
        for (int i = 0; i < TILE; i++)
            for (int j = 0; j < TILE; j++)
                B[(TILE*thid_y+i)*TILE*m+TILE*thid_x+j] = 
                A[(TILE*thid_x+j)*TILE*m+TILE*thid_y+i];
}

#include <stdio.h>
__global__
void transpose_kernel_register(float4 * A, float4 * B, size_t m) {

    const size_t thid_x = blockDim.x*blockIdx.x+threadIdx.x,
                 thid_y = blockDim.y*blockIdx.y+threadIdx.y;

    auto swap = [] (float& lhs, float& rhs) {
        float tmp = lhs; lhs = rhs; rhs = tmp;
    };

    if (thid_x < m && thid_y < m) {

        float4 row0 = A[thid_x*m*4+thid_y];
        float4 row1 = A[(thid_x*m*4+1*m)+thid_y];
        float4 row2 = A[(thid_x*m*4+2*m)+thid_y];
        float4 row3 = A[(thid_x*m*4+3*m)+thid_y];

        swap(row0.y, row1.x);
        swap(row0.z, row2.x);
        swap(row0.w, row3.x);
        swap(row1.z, row2.y);
        swap(row1.w, row3.y);
        swap(row2.w, row3.z);
        
        B[thid_y*m*4+thid_x]       = row0;
        B[(thid_y*m*4+1*m)+thid_x] = row1;
        B[(thid_y*m*4+2*m)+thid_x] = row2;
        B[(thid_y*m*4+3*m)+thid_x] = row3;
    }
}

int main () {
    
    // use the first GPU (0..Tesla K40 12G RAM, 1..Titan 6G RAM)
    cudaSetDevice(0);                                                     CUERR

    // small letters for host, capital letters for device memory
    float *a, *b, *A, *B;

    // allocate host and device memory
    cudaMallocHost(&a, sizeof(float)*N*N);                                CUERR
    cudaMallocHost(&b, sizeof(float)*N*N);                                CUERR
    cudaMalloc(&A, sizeof(float)*N*N);                                    CUERR
    cudaMalloc(&B, sizeof(float)*N*N);                                    CUERR

    // fill a with ascending numbers
    TIMERSTART(fillArraysOnHostSide)
    std::iota(a, a+N*N, 0);             // (0, 1, 2, 3, ..., N-1)
    TIMERSTOP(fillArraysOnHostSide)

    // measure the time for overall execution on GPU
    TIMERSTART(overallCUDA)

    // copy v to V from host to device
    TIMERSTART(copyHostToDevice)
    cudaMemcpy(A, a, sizeof(float)*N*N, cudaMemcpyHostToDevice);          CUERR
    cudaMemset(B, 0, sizeof(float)*N*N);                                  CUERR
    TIMERSTOP(copyHostToDevice)

    // define appropriate grid and block parameters here
    dim3 blocksPerGrid(SDIV(N, 32), SDIV(N, 32));
    dim3 threadsPerBlock(32, 32);

    // invoke the kernel
    TIMERSTART(kernelTime)
    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);        CUERR
    TIMERSTOP(kernelTime)

    // define appropriate grid and block parameters here
    dim3 blocksPerGridTile(SDIV(N/TILE, 8), SDIV(N/TILE, 8));
    dim3 threadsPerBlockTile(8, 8);

    // invoke the kernel
    TIMERSTART(kernelTimeTiled)
    transpose_kernel_tiled<<<blocksPerGridTile, threadsPerBlockTile>>>
                          (A, B, N/TILE);                                 CUERR
    TIMERSTOP(kernelTimeTiled)

    // define appropriate grid and block parameters here
    dim3 blocksPerGridReg(SDIV(N/4, 8), SDIV(N/4, 8));
    dim3 threadsPerBlockReg(8, 8);

    // invoke the kernel
    TIMERSTART(kernelTimeRegister)
    transpose_kernel_register<<<blocksPerGridReg, threadsPerBlockReg>>>
                          ((float4*)A, (float4*)B, N/4);                  CUERR
    TIMERSTOP(kernelTimeRegister)


    // copy V to v from device to host
    TIMERSTART(copyDeviceToHost)
    cudaMemcpy(b, B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);          CUERR
    TIMERSTOP(copyDeviceToHost)

    // stop overall GPU timer and print result
    TIMERSTOP(overallCUDA)

    ///////////////////////////////////////////////////////////////////////////
    // BENCHMARKS AND CHECKS (you may ignore this, especially the openMP part)
    ///////////////////////////////////////////////////////////////////////////

    // check for correct result computed by CUDA
    bool going_on = true;
    for (size_t i = 0; i < N && going_on; i++)
        for (size_t j = 0; j < N && going_on; j++)
            if (b[i*N+j] != a[j*N+i]) {
                std::cout << "error at position (" 
                          << i << "," << j << ")" << std::endl;
                going_on = false;
            }

    // measure time on single-threaded host
    TIMERSTART(overallSingleCore)
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            b[i*N+j] = a[j*N+i];
    TIMERSTOP(overallSingleCore)

    // measure time on multi-threaded host
    TIMERSTART(overallMultiCore)
    # pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            b[i*N+j] = a[j*N+i];
    TIMERSTOP(overallMultiCore)

    // get rid of the memory
    cudaFree(A);
    cudaFree(B);
    cudaFreeHost(a);
    cudaFreeHost(b);

    // print status
    float usedMem = 2*N*N*sizeof(float)/(1L<<30);
    std::cout << "#processed " << usedMem << " gigabytes." << std::endl;
    std::cout << "CUDA programming is fun!" << std::endl;
}
