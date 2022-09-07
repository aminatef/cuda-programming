#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include"utils.h"
using std::cout;
using std::endl;
using std ::generate;
using std ::vector;

//OPTIMIZATION 4 
#define TSM 64 // TILE SIZE IN M DIM
#define TSN 64 // TILE SIZE IN N DIM
#define TSK 32 // TILE SIZE IN K DIM
#define WPTN 8 // WORK PER THREAD IN N DIM
#define WPTM 1// WORK PER THREAD IN M DIM
#define RTSN (TSN/WPTN) // REDUCED TILE SIZE IN N DIM
#define RTSM (TSM/WPTM) // REDUCED TILE SIZE IN M DIM
#define LPT ((TSK*TSM)/(RTSM*RTSN)) // LOADS PER THREAD

//PARAMTERS OF TRANSPOSE KERNEL
#define TILE_DIM 32 
#define BLOCK_ROWS 32

//KERNEL CONFIG
const int numThreads = 64;
const int numElements = 1 << 7; 

__global__ void GEMM_opt4(int M, int N, int K, const float *A, const float *B, float *C){
    // all matrix are stored in col major format
    // B is transposed 
    // padding to reduce memory bank conflicts
    // fully utilizing shared memory by using rectangler tiles
    // A DIM -> k*M
    // B DIM -> K*N
    // C DIM -> M*N


    const int row = threadIdx.x;
    const int col = threadIdx.y;

    const int globalRow = TSM * blockIdx.x + row;
    const int globalCol = TSN * blockIdx.y + col;

    __shared__ float Asub[TSK][TSM];
    __shared__ float Bsub[TSN][TSK+2]; // +2 ->PADDING TO AVOID MEMORY BANK CONFLICTS

    //INIT ACCUMALTE ARRAY
    float acc[WPTN];
    for(int w = 0 ;w<WPTN;w++){acc[w]=0;}

    int numTiles = K/TSK;

    for(int i = 0 ; i<numTiles;i++){
        for(int l=0;l<LPT;l++){
            int tiledIndex = i * TSK + col + l*RTSN;
            int indexA  = tiledIndex*M + blockIdx.x*TSM + row;
            int indexB  = tiledIndex*N + blockIdx.y*TSN + row;
            Asub[col+l*RTSN][row] = A[indexA];
            Bsub[row][col+l*RTSN] = B[indexB];
        }
        __syncthreads();

        for(int j = 0 ; j < TSK ; j++){
            for(int w = 0 ; w < WPTN ; w++){
                acc[w] += Asub[j][row] * Bsub[col+w*RTSN][j];
            }
        }
        __syncthreads();
    }
    
    for(int i = 0 ;i < WPTN;i++){
        C[(globalCol+i*RTSN)*M+globalRow]=acc[i];
    }
}


__global__ void transposeNaive(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

void test_opt4(){
    size_t size = sizeof(float) * numElements * numElements;
    vector<float> h_a(numElements * numElements);
    vector<float> h_b(numElements * numElements);
    vector<float> h_c(numElements * numElements);
    vector<float> test_data(numElements * numElements);

    generate(h_a.begin(), h_a.end(), RandomNumber);
    generate(h_b.begin(), h_b.end(), RandomNumber);

    // device memory allocation
    float *device_A = NULL;
    cudaMalloc((void **)&device_A, size);

    float *device_B = NULL;
    cudaMalloc((void **)&device_B, size);

    float *device_T = NULL;
    cudaMalloc((void **)&device_T, size);

    float *device_C = NULL;
    cudaMalloc((void **)&device_C, size);

    cudaMemcpy(device_A, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, h_b.data(), size, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY= 32 ;
    dim3 block(blockX,blockY);
    dim3 grid((numElements+blockX-1)/blockX , (numElements+blockY-1)/blockY);

    // for(int i=0;i<numElements;i++){
    //     for(int j = 0;j<numElements;j++){
    //         cout<<h_a[i*numElements+j]<<"  ";
    //     }
    //     cout<<endl;
    // }
    cout<<"========================="<<endl;
    transposeNaive<<<grid,block>>>(device_T,device_A);
    cudaMemcpy(test_data.data(), device_T, size, cudaMemcpyDeviceToHost);
    // for(int i=0;i<numElements;i++){
    //     for(int j = 0;j<numElements;j++){
    //         cout<<test_data[i*numElements+j]<<"  ";
    //     }
    //     cout<<endl;
    // }

    int numBlocks = (numElements + numThreads - 1) / numThreads;
    dim3 threads(numThreads, numThreads/WPTN);
    dim3 blocks(numBlocks, numBlocks);
    GEMM_opt4<<<blocks, threads>>>(numElements, numElements, numElements, device_B, device_T, device_C);
    cudaMemcpy(h_c.data(), device_C, size, cudaMemcpyDeviceToHost);
    verify_result(h_a, h_b, h_c,numElements);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

}
int main()
{
    test_opt4();
}


