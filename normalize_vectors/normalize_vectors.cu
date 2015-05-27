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
// compile with: nvcc normalize_vectors.cu -std=c++11 -O3 -arch=sm_30 \
//               -Xcompiler="-fopenmp" -o normalize_vectors
///////////////////////////////////////////////////////////////////////////////

// 4 gigabytes of memory for float4 (device 0 has 12G RAM in total)
#define N (1L<<28)

__global__
void norm_kernel(float * V, size_t n) {

    size_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    if (thid < n) {
        const float x = V[4*thid],
                    y = V[4*thid+1],
                    z = V[4*thid+2],
                    w = V[4*thid+3];

        const float rev_sqrt = rsqrtf(x*x+y*y+z*z+w*w);
        
        V[4*thid]   = x*rev_sqrt;
        V[4*thid+1] = y*rev_sqrt;
        V[4*thid+2] = z*rev_sqrt;
        V[4*thid+3] = w*rev_sqrt;
   }
}

__global__
void norm_kernel_packed(float4 * V, size_t n) {

    size_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    if (thid < n) {
        float4 s = V[thid];
        const float rev_sqrt = rsqrtf(s.x*s.x+s.y*s.y+s.z*s.z+s.w*s.w);
        
        s.x *= rev_sqrt;
        s.y *= rev_sqrt;
        s.z *= rev_sqrt;
        s.w *= rev_sqrt;

        V[thid] = s;
   }
}

int main () {
    
    // use the first GPU (0..Tesla K40 12G RAM, 1..Titan 6G RAM)
    cudaSetDevice(0);                                                     CUERR

    // small letters for host, capital letters for device memory
    float *v, *V;

    // allocate host and device memory
    cudaMallocHost(&v, sizeof(float)*4*N);                                CUERR
    cudaMalloc(&V, sizeof(float)*4*N);                                    CUERR

    // fill a and b with stuff
    TIMERSTART(fillArraysOnHostSide)
    std::iota(v, v+4*N, 0);             // (0, 1, 2, 3, ..., N-1)
    TIMERSTOP(fillArraysOnHostSide)

    // measure the time for overall execution on GPU
    TIMERSTART(overallCUDA)

    // copy v to V from host to device
    TIMERSTART(copyHostToDevice)
    cudaMemcpy(V, v, sizeof(float)*4*N, cudaMemcpyHostToDevice);          CUERR
    TIMERSTOP(copyHostToDevice)

    // invoke the kernel
    TIMERSTART(kernelTime)
    norm_kernel<<<SDIV(N, 1024), 1024>>>(V, N);                           CUERR
    TIMERSTOP(kernelTime)

    // invoke the kernel
    TIMERSTART(kernelTimePacked)
    norm_kernel_packed<<<SDIV(N, 1024), 1024>>>((float4*)V, N);           CUERR
    TIMERSTOP(kernelTimePacked)

    // copy V to v from device to host
    TIMERSTART(copyDeviceToHost)
    cudaMemcpy(v, V, sizeof(float)*4*N, cudaMemcpyDeviceToHost);          CUERR
    TIMERSTOP(copyDeviceToHost)

    // stop overall GPU timer and print result
    TIMERSTOP(overallCUDA)

    ///////////////////////////////////////////////////////////////////////////
    // BENCHMARKS AND CHECKS (you may ignore this, especially the openMP part)
    ///////////////////////////////////////////////////////////////////////////

    // check for correct result computed by CUDA
    for (size_t index = 0; index < N; index++) {
        const float x = v[4*index],
                    y = v[4*index+1],
                    z = v[4*index+2],
                    w = v[4*index+3];
        const float  residue = x*x+y*y+z*z+w*w-1;
        if (residue*residue > 1E-6) {
            std::cout << "error at postion " << index << std::endl;
            break;
        }
    }

    // measure time  on single-threaded host
    TIMERSTART(overallSingleCore)
    for (size_t index = 0; index < N; index++) {
        const float x = v[4*index],
                    y = v[4*index+1],
                    z = v[4*index+2],
                    w = v[4*index+3];

        const float rev_sqrt = 1.0/std::sqrt(x*x+y*y+z*z+w*w);
        
        v[4*index]   = x*rev_sqrt;
        v[4*index+1] = y*rev_sqrt;
        v[4*index+2] = z*rev_sqrt;
        v[4*index+3] = w*rev_sqrt;
    }
    TIMERSTOP(overallSingleCore)

    // measure time on multi-threaded host
    TIMERSTART(overallMultiCore)
    # pragma omp parallel for
    for (size_t index = 0; index < N; index++) {
        const float x = v[4*index],
                    y = v[4*index+1],
                    z = v[4*index+2],
                    w = v[4*index+3];

        const float rev_sqrt = 1.0/std::sqrt(x*x+y*y+z*z+w*w);
        
        v[4*index]   = x*rev_sqrt;
        v[4*index+1] = y*rev_sqrt;
        v[4*index+2] = z*rev_sqrt;
        v[4*index+3] = w*rev_sqrt;
    }
    TIMERSTOP(overallMultiCore)

    // get rid of the memory
    cudaFree(V);
    cudaFreeHost(v);

    // print status
    float usedMem = 4.0*N*sizeof(float)/(1L<<30);
    std::cout << "#processed " << usedMem << " gigabytes." << std::endl;
    std::cout << "CUDA programming is fun!" << std::endl;
}
