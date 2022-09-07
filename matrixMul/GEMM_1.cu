#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include"verifyResult.h"
using std::cout;
using std::endl;
using std ::generate;
using std ::vector;

//OPTIMIZATION 1 TILE SIZE
#define TILE_SIZE  32

//KERNEL CONFIG
const int numThreads = 32;
const int numElements = 1 << 7; 

int RandomNumber()
{
    return (rand() % 100);
}

__global__ void GEMM_opt1(int M, int N, int K, const float *A, const float *B, float *C)
{

    // all matrix are stored in col major format
    // A DIM -> k*M
    // B DIM -> N*k
    // C DIM -> M*N

    // reversed indexes for storing result in C
    int global_row = threadIdx.x + blockIdx.x * TILE_SIZE;
    int global_col = threadIdx.y + blockIdx.y * TILE_SIZE;

    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int numTiles = K / TILE_SIZE;
    float acc = 0;
    for (int i = 0; i < numTiles; i++)
    {
        int tiledRow = i * TILE_SIZE + threadIdx.x;
        int tiledCol = i * TILE_SIZE + threadIdx.y;
        Asub[threadIdx.y][threadIdx.x] = A[global_row + tiledCol * M];
        Bsub[threadIdx.y][threadIdx.x] = B[global_col * K + tiledRow];
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++)
        {
            acc += Asub[j][threadIdx.x] * Bsub[threadIdx.y][j];
        }
        __syncthreads();
    }
    C[global_col * M + global_row] = acc;
}

void test_opt1(){
    size_t size = sizeof(float) * numElements * numElements;
    vector<float> h_a(numElements * numElements);
    vector<float> h_b(numElements * numElements);
    vector<float> h_c(numElements * numElements);

    generate(h_a.begin(), h_a.end(), RandomNumber);
    generate(h_b.begin(), h_b.end(), RandomNumber);

    // device memory allocation
    float *device_A = NULL;
    cudaMalloc((void **)&device_A, size);

    float *device_B = NULL;
    cudaMalloc((void **)&device_B, size);

    float *device_C = NULL;
    cudaMalloc((void **)&device_C, size);

    cudaMemcpy(device_A, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, h_b.data(), size, cudaMemcpyHostToDevice);

    int numBlocks = (numElements + numThreads - 1) / numThreads;
    dim3 threads(numThreads, numThreads);
    dim3 blocks(numBlocks, numBlocks);
    GEMM_opt1<<<blocks, threads>>>(numElements, numElements, numElements, device_B, device_A, device_C);
    cudaMemcpy(h_c.data(), device_C, size, cudaMemcpyDeviceToHost);
    verify_result(h_a, h_b, h_c,numElements);

    cout << "COMPLETED SUCCESSFULLY\n";



    // Free memory on device
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}
int main(){
    test_opt1();
}



