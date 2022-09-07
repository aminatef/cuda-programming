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

//OPTIMIZATION 3 WIDE DATA LOAD
#define WIDTH  8 // wide data load width
#define TILE_SIZE  32 //TILE SIZE

//KERNEL CONFIG
const int numThreads = 32;
const int numElements = 1 << 7; 

struct  float8
{
    float s0,s1, s2, s3, s4, s5, s6, s7;
    __host__ __device__ inline float8 operator =(float8 a){
        s0 = a.s0;
        s1 = a.s1;
        s2 = a.s2;
        s3 = a.s3;
        s4 = a.s4;
        s5 = a.s5;
        s6 = a.s6;
        s7 = a.s7;
        return a;
    }
};

float8 RandomNumber_8()
{
    float8 f;
    f.s0=rand()%100;f.s1=rand()%100;f.s2=rand()%100;f.s3=rand()%100;
    f.s4=rand()%100;f.s5=rand()%100;f.s6=rand()%100;f.s7=rand()%100;
    return f;
}

vector<float> transform_float8_2float(const vector<float8> &a,int n,int k){
    vector<float> res(n*k);
    float8 tmp;
    for(int i = 0 ;i<n/WIDTH;i++){
        for(int j = 0 ;j<k;j++){
            tmp = a[i*k+j];
            //cout<<tmp.s0<<" "<<tmp.s1<<" "<<tmp.s2<<" "<<tmp.s3<<" "<<tmp.s4<<" "<<tmp.s5<<" "<<tmp.s6<<" "<<tmp.s7<<endl;
            // res[i*WIDTH*n+j]     = tmp.s0;
            // res[(i*WIDTH+1)*n+j] = tmp.s1;
            // res[(i*WIDTH+2)*n+j] = tmp.s2;
            // res[(i*WIDTH+3)*n+j] = tmp.s3;
            // res[(i*WIDTH+4)*n+j] = tmp.s4;
            // res[(i*WIDTH+5)*n+j] = tmp.s5;
            // res[(i*WIDTH+6)*n+j] = tmp.s6;
            // res[(i*WIDTH+7)*n+j] = tmp.s7;

            res[i*n+j*WIDTH]   = tmp.s0;
            res[i*n+j*WIDTH+1] = tmp.s1;
            res[i*n+j*WIDTH+2] = tmp.s2;
            res[i*n+j*WIDTH+3] = tmp.s3;
            res[i*n+j*WIDTH+4] = tmp.s4;
            res[i*n+j*WIDTH+5] = tmp.s5;
            res[i*n+j*WIDTH+6] = tmp.s6;
            res[i*n+j*WIDTH+7] = tmp.s7;  
        }
    }
    return res;
}

__global__ void GEMM_opt3(int M, int N, int K, const float8 *A, const float8 *B, float8 *C)
{
    // all matrix are stored in col major format
    // Wider loads
    // A DIM -> k*M
    // B DIM -> N*k
    // C DIM -> M*N


    int col = threadIdx.y;
    int row = threadIdx.x;

    int globalRow = threadIdx.x + blockDim.x*blockIdx.x;
    int globalCol = threadIdx.y + blockDim.y*blockIdx.y;

    __shared__ float8 Asub[TILE_SIZE][TILE_SIZE/WIDTH];
    __shared__ float8 Bsub[TILE_SIZE][TILE_SIZE/WIDTH];
    // if(globalRow==0 && globalCol==0){
    //     printf("%f,%f,%f,%f,%f,%f,%f,%f",A[0].s0,A[0].s1,A[0].s2,A[0].s3,A[0].s4,A[0].s5,A[0].s6,A[0].s7);
    // }
    float8 acc  = {0,0,0,0,0,0,0,0};
    float8 vecA = {0,0,0,0,0,0,0,0};
    float8 vecB = {0,0,0,0,0,0,0,0};
    // if(globalRow==0 && globalCol==0)
    //     printf("%f",acc.s0);
    float valB=0;
    int numTiles = K / TILE_SIZE;
    // if(globalRow == 0 && globalCol == 0){
    //         printf("%f",vecA.s0);
    //         printf("%f",vecB.s0);
    //         printf("%d\n",numTiles);
    // }

    for(int i = 0 ; i < numTiles ; i++){
        int tiledCol = i * TILE_SIZE + col;
        int tiledRow = i * TILE_SIZE/WIDTH + row;

        Asub[col][row] = A[(tiledCol)*(M/WIDTH)+globalRow];
        Bsub[col][row] = B[(tiledRow)+globalCol*(K/WIDTH)];
        // if(globalRow == 0 && globalCol == 0){
        //     printf("%f == %f \n",Asub[col][row].s0,A[(tiledCol)*(M/WIDTH)+globalRow].s0);
        //     printf("%f == %f",Bsub[col][row].s0,B[(tiledRow)+globalCol*(K/WIDTH)].s0);
        // }

        __syncthreads();
        for(int j = 0 ;j<TILE_SIZE/WIDTH;j++){
            vecB = Bsub[col][j];
            for(int w = 0 ;w<WIDTH;w++)
            {
                vecA = Asub[WIDTH*j+w][row];
                switch(w){
                    case 0:valB = vecB.s0; break; case 1:valB = vecB.s1; break;
                    case 2:valB = vecB.s2; break; case 3:valB = vecB.s3; break;
                    case 4:valB = vecB.s4; break; case 5:valB = vecB.s5; break;
                    case 6:valB = vecB.s6; break; case 7:valB = vecB.s7; break;
                }

                acc.s0 += vecA.s0 * valB; acc.s1 += vecA.s1 * valB;
                acc.s2 += vecA.s2 * valB; acc.s3 += vecA.s3 * valB;
                acc.s4 += vecA.s4 * valB; acc.s5 += vecA.s5 * valB;
                acc.s6 += vecA.s6 * valB; acc.s7 += vecA.s7 * valB;

            }
        }
        __syncthreads();
    }
    // if(globalRow==0 && globalCol==0)
    //     printf("%f",C[(globalCol*M/WIDTH)+globalRow].s0);
    //printf("%f,%f,%f,%f,%f,%f,%f,%f",acc.s0,acc.s1,acc.s2,acc.s3,acc.s4,acc.s5,acc.s6,acc.s7);
    C[(globalCol*M/WIDTH)+globalRow] = acc;
}

void test_opt3(){
    
    size_t size = sizeof(float8) * (numElements/WIDTH) * numElements;
    vector<float8> h_a(numElements * numElements/WIDTH);
    vector<float8> h_b(numElements * numElements/WIDTH);
    vector<float8> h_c(numElements * numElements/WIDTH);

    generate(h_a.begin(), h_a.end(), RandomNumber_8);
    generate(h_b.begin(), h_b.end(), RandomNumber_8);

    // device memory allocation
    float8 *device_A = NULL;
    cudaMalloc((void **)&device_A, size);

    float8 *device_B = NULL;
    cudaMalloc((void **)&device_B, size);

    float8 *device_C = NULL;
    cudaMalloc((void **)&device_C, size);

    cudaMemcpy(device_A, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, h_b.data(), size, cudaMemcpyHostToDevice);

    int numBlocks = (numElements + numThreads - 1) / numThreads;
    dim3 threads(numThreads/WIDTH, numThreads);
    dim3 blocks(numBlocks, numBlocks);
    GEMM_opt3<<<blocks, threads>>>(numElements, numElements, numElements, device_B, device_A, device_C);
    cudaMemcpy(h_c.data(), device_C, size, cudaMemcpyDeviceToHost);
    vector<float> test = transform_float8_2float(h_b,numElements,numElements);
    cout<<"==========================="<<endl;;
    // for(int i=0;i<numElements;i++){
    //     for(int j = 0;j<numElements;j++){
    //         cout<<test[i*numElements+j]<<"  ";
    //     }
    //     cout<<endl;
    // }
    verify_result(transform_float8_2float(h_a,numElements,numElements),
                 transform_float8_2float(h_b,numElements,numElements),
                 transform_float8_2float(h_c,numElements,numElements),numElements);

    cout << "COMPLETED SUCCESSFULLY\n";



    // Free memory on device
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C); 
}
int main(){
    test_opt3();
}

