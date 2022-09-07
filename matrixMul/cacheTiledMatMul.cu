#include<vector>
#include<algorithm>
#include <cstdlib>
#include<iostream>
#include<assert.h>
using std ::vector;
using std ::generate;
using std:: cout;
int RandomNumber(){
    return (rand()%100);
}
const int N = 1<<10;
const int SHMEM_SIZE = 1<<10;

__global__ void matrixMul(const int * a ,const int *b ,int *c){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    int tmp = 0;
    for(int i = 0;i<N;i+=blockDim.x){
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N+i+threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i*N + threadIdx.y*N + col];
        __syncthreads();

        for (int j = 0;j<blockDim.x;j++){
            tmp += s_a[threadIdx.y*blockDim.x+j] * s_b[j*blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    c[row*N+col]=tmp;
    
    
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}


int main (){
    cudaError_t error = cudaSuccess;
    int numElements = 1024;
    size_t size =sizeof(int)*numElements*numElements;
    vector<int> h_a(numElements*numElements);
    vector<int> h_b(numElements*numElements);
    vector<int> h_c(numElements*numElements);

    generate(h_a.begin(),h_a.end(),RandomNumber);
    generate(h_b.begin(),h_b.end(),RandomNumber);


    // device memory allocation
    int *device_A = NULL;
    error = cudaMalloc((void **)&device_A,size);
    if (error != cudaSuccess){
        fprintf(stderr,"Falied to allocate device vector A\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    int *device_B = NULL;
    error = cudaMalloc((void **)&device_B,size);
    if (error != cudaSuccess){
        fprintf(stderr,"Falied to allocate device vector B\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    int *device_C = NULL;
    error = cudaMalloc((void **)&device_C,size);
    if (error != cudaSuccess){
        fprintf(stderr,"Falied to allocate device vector C\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error  = cudaMemcpy(device_A,h_a.data(),size,cudaMemcpyHostToDevice);
    error  = cudaMemcpy(device_B,h_b.data(),size,cudaMemcpyHostToDevice);
    int numThreads = 32;

    int numBlocks = (numElements+numThreads-1)/numThreads;
    dim3 threads(numThreads,numThreads);
    dim3 blocks(numBlocks,numBlocks);
    matrixMul<<<blocks,threads>>>(device_A,device_B,device_C);
    cudaMemcpy(h_c.data(),device_C,size,cudaMemcpyDeviceToHost);
    verify_result(h_a, h_b, h_c);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

  return 0;


}
