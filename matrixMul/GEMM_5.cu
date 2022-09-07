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

//OPTIMIZATION 5
#define TSM 128 // TILE SIZE IN M DIM
#define TSN 128 // TILE SIZE IN N DIM
#define TSK 16 // TILE SIZE IN K DIM
#define WPTN 8 // WORK PER THREAD IN N DIM
#define WPTM 8// WORK PER THREAD IN M DIM
#define RTSN (TSN/WPTN) // REDUCED TILE SIZE IN N DIM
#define RTSM (TSM/WPTM) // REDUCED TILE SIZE IN M DIM
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // LOADS PER THREAD
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // LOADS PER THREAD

//KERNEL CONFIG
const int numThreads = 128;
const int numElements = 1 << 7; 

//PARAMTERS OF TRANSPOSE KERNEL
#define TILE_DIM 32 
#define BLOCK_ROWS 32

__global__ void transposeNaive(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

__global__ void GEMM_opt5(int M, int N, int K, const float *A, const float *B, float *C){
    // all matrix are stored in col major format
    // B is transposed 
    // padding to reduce memory bank conflicts
    // fully utilizing shared memory by using rectangler tiles
    // 2d register blocking
    // A DIM -> k*M
    // B DIM -> K*N
    // C DIM -> M*N 

    int tid_n = threadIdx.y;
    int tid_m = threadIdx.x;

    int offset_m = blockIdx.x * TSM;
    int offset_n = blockIdx.y * TSN;

    __shared__ float Asub[TSK][TSM];
    __shared__ float Bsub[TSN][TSK+2];

    
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];
    for(int i =0;i<WPTM;i++)
        for(int j = 0;j<WPTN;j++)
            acc[i][j]=0;
    
    int numTiles = K / TSK;

    for(int i = 0 ;i < numTiles ; i++){
        for(int l = 0 ; l < LPTA;l++){
            int tid = tid_n * RTSM + tid_m; 

            int id = tid + RTSM*RTSN*l;

            int row = id%TSM;
            int col = id/TSM;
            
            int tiledIndex = (i*TSK + col);

            int indexA = tiledIndex*M+offset_m+row;
            int indexB = tiledIndex*N+offset_n+row;

            Asub[col][row] = A[indexA];
            Bsub[row][col] = B[indexB];

        }
        __syncthreads();
        for(int t = 0 ;t<TSK;t++){
            for(int w = 0 ; w<WPTN;w++){
                Breg[w]=Bsub[tid_n+w*RTSN][t];
            }
            for(int wm = 0 ;wm<WPTM;wm++){
                Areg = Asub[t][tid_m+wm*RTSM];
                for(int wn = 0 ; wn<WPTN; wn++){
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
        __syncthreads();
    }
    for(int wm = 0 ;wm<WPTM;wm++){
        int globalRow = offset_m +tid_m+wm*RTSM;
        for(int wn = 0 ; wn<WPTN; wn++){
            int globalCol = offset_n + tid_n+wn*RTSN;
            C[globalCol*M+globalRow]=acc[wm][wn];
        }
    }
}

void test_opt5(){
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

    transposeNaive<<<grid,block>>>(device_T,device_A);
    cudaMemcpy(test_data.data(), device_T, size, cudaMemcpyDeviceToHost);

    int numBlocks = (numElements + numThreads - 1) / numThreads;
    dim3 threads(numThreads/WPTM, numThreads/WPTN);
    dim3 blocks(numBlocks, numBlocks);
    GEMM_opt5<<<blocks, threads>>>(numElements, numElements, numElements, device_B, device_T, device_C);
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
    test_opt5();
    return 0;
}