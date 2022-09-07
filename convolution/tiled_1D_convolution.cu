#include<algorithm>
#include<cassert>
#include<iostream>
#include<vector>
#define MASK_SIZE 7 
__constant__ int const_mask[MASK_SIZE];
__global__ void conv_1D_tiled(int * d_a,int *d_result,int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    extern __shared__ int s_array[];

    int r = MASK_SIZE /2;
    int d = r*2;

    s_array[threadIdx.x] = d_a[idx];

    int offset = threadIdx.x+blockDim.x;
    if(offset<blockDim.x+d){
        s_array[offset] = d_a[blockIdx.x*blockDim.x+offset];
    }
    __syncthreads();
    int tmp=0;
    for (int i = 0 ; i < MASK_SIZE;i++){
        tmp += s_array[threadIdx.x+i]*const_mask[i];
    }
    d_result[idx]=tmp;
}


void verify_result(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < MASK_SIZE; j++) {
      temp += array[i + j] * mask[j];
    }
    assert(temp == result[i]);
  }
}


int main(){
    int N = 1<<16;
    size_t bytes = N*sizeof(int);
    size_t bytes_m = MASK_SIZE*sizeof(int);
    //radius for padding the array 
    int r = MASK_SIZE /2;
    int n_p = N + r*2;

    int * h_a = (int*)malloc(n_p*sizeof(int));
    
    int * h_result = (int*)malloc(bytes);

    int *h_mask = (int*)malloc(bytes_m);

    for(int i = 0; i<n_p;i++){
        if((i<r)||i>=(N+r)){
            h_a[i] = 0;
        }else{
            h_a[i] = rand()%100;
        }
    }
    for(int i = 0; i<MASK_SIZE;i++){
        h_mask[i] = rand()%10;
    }

    int*d_a,*d_result;
    cudaMalloc((void**)&d_a,n_p*sizeof(int));
    cudaMalloc((void**)&d_result,bytes);

    cudaMemcpy(d_a,h_a,n_p*sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(const_mask,h_mask,bytes_m);

    int numthreads = 256;
    int numblocks = (N+numthreads - 1)/numthreads;

    size_t SHMEM = (numthreads+2*r)*sizeof(int);

    conv_1D_tiled<<<numblocks,numthreads,SHMEM>>>(d_a,d_result,N);
    cudaMemcpy(h_result,d_result,bytes,cudaMemcpyDeviceToHost);
    verify_result(h_a,h_mask, h_result, N);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    // Free allocated memory on the device and host
    cudaFree(d_result);
    cudaFree(d_a);
}