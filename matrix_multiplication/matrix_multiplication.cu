#include<iostream>    // cout, endl
#include<omp.h>       // benchmark below (mutli-threading with openMP pragmas)

///////////////////////////////////////////////////////////////////////////////
// IGNORE THESE HELPERS (taken from https://github.com/gravitino/cudahelpers)
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
// compile with: nvcc matrix_multiplication.cu -std=c++11 -O3 -arch=sm_30 \
//               -Xcompiler="-fopenmp" -o matrix_multiplication
///////////////////////////////////////////////////////////////////////////////

#define L (1L<<10)
#define TILE_WIDTH (32)

__global__
void MatrixMulKernel(float * Md, float * Nd, float * Pd, int Width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int m = 0; m < Width/TILE_WIDTH; m++) {
        Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH +ty)*Width];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }

    Pd[Row*Width+Col] = Pvalue;
}

int main () {
    
    // use the first GPU (0..Tesla K40 12G RAM, 1..Titan 6G RAM)
    cudaSetDevice(0);                                                     CUERR

    float *M, *N, *P, *Md, *Nd, *Pd;

    // allocate host and device memory
    cudaMallocHost(&M, sizeof(float)*L*L);                                CUERR
    cudaMallocHost(&N, sizeof(float)*L*L);                                CUERR
    cudaMallocHost(&P, sizeof(float)*L*L);                                CUERR
    cudaMalloc(&Md, sizeof(float)*L*L);                                   CUERR
    cudaMalloc(&Nd, sizeof(float)*L*L);                                   CUERR
    cudaMalloc(&Pd, sizeof(float)*L*L);                                   CUERR

    TIMERSTART(fillMatricesOnHostSide)
    for (size_t i = 0; i < L; i++)
        for (size_t j = 0; j < L; j++) {
            M[i*L+j] = (i+j) % 3;
            N[i*L+j] = (j+i) % 2;
        }
    TIMERSTOP(fillMatricesOnHostSide)

    TIMERSTART(overallCUDA)

    TIMERSTART(copyHostToDevice)
    cudaMemcpy(Md, M, sizeof(float)*L*L, cudaMemcpyHostToDevice);         CUERR
    cudaMemcpy(Nd, N, sizeof(float)*L*L, cudaMemcpyHostToDevice);         CUERR
    cudaMemset(Pd, 0, sizeof(float)*L*L);                                 CUERR
    TIMERSTOP(copyHostToDevice)

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(SDIV(L, TILE_WIDTH), SDIV(L, TILE_WIDTH));
    TIMERSTART(Kernel)
    MatrixMulKernel<<<blocksPerGrid,threadsPerBlock>>>(Md, Nd, Pd, L);    CUERR
    TIMERSTOP(Kernel)

    TIMERSTART(copyDeviceToHost)
    cudaMemcpy(P, Pd, sizeof(float)*L*L, cudaMemcpyDeviceToHost);         CUERR
    TIMERSTOP(copyDeviceToHost)

    TIMERSTOP(overallCUDA)

    ///////////////////////////////////////////////////////////////////////////
    // BENCHMARKS AND CHECKS (you may ignore this, especially the openMP part)
    ///////////////////////////////////////////////////////////////////////////

    // check for correct result computed by CUDA
    bool going_on = true;
    for (size_t i = 0; i < L && going_on; i++)
        for (size_t j = 0; j < L && going_on; j++) {
            float value = 0;
            for (size_t k = 0; k < L; k++)
                value += M[i*L+k] * N[k*L+j];
            if (P[i*L+j] != value) {
                std::cout << value << " " << P[i*L+j] << std::endl;
                std::cout << "error at position (" << i
                          << "," << j << ")" << std::endl;
                going_on = false;
            }
        }
 
    // measure time on single-threaded host
    TIMERSTART(overallSingleCore)
    for (size_t i = 0; i < L; i++)
        for (size_t j = 0; j < L; j++) {
            float value = 0;
            for (size_t k = 0; k < L; k++)
                value += M[i*L+k] * N[k*L+j];
            P[i*L+j] = value;
        }
    TIMERSTOP(overallSingleCore)

    // measure time on multi-threaded host
    TIMERSTART(overallMultiCore)
    # pragma omp parallel for 
    for (size_t i = 0; i < L; i++)
        for (size_t j = 0; j < L; j++) {
            float value = 0;
            for (size_t k = 0; k < L; k++)
                value += M[i*L+k] * N[k*L+j];
            P[i*L+j] = value;
        }
    TIMERSTOP(overallMultiCore)

    // get rid of the memory
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
    cudaFreeHost(M);
    cudaFreeHost(N);
    cudaFreeHost(P);

    // print status
    float usedMem = 3.0*L*L*sizeof(float)/(1L<<30);
    std::cout << "#processed " << usedMem << " gigabytes." << std::endl;
    std::cout << "CUDA programming is fun!" << std::endl;
}
