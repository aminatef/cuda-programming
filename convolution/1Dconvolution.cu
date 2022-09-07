#include<algorithm>
#include<cassert>
#include<iostream>
#include<vector>
#define MASK_SIZE 7 
__constant__ int const_mask[MASK_SIZE];

__global__ void conv_1D(int * d_a,int *d_result,int *d_mask,int n,int m){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int r = m /2;
    int start = idx -r;
    int tmp=0;
    for (int i = 0 ; i < m;i++){
        if( ((start+i)>=0) && ((start+i)<n) ){
            tmp += d_a[start+i]*d_mask[i];
        }
    }
    d_result[idx]=tmp;

}

__global__ void conv_1D_constant_memory(int * d_a,int *d_result,int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int r = MASK_SIZE /2;
    int start = idx -r;
    int tmp=0;
    for (int i = 0 ; i < MASK_SIZE;i++){
        if( ((start+i)>=0) && ((start+i)<n) ){
            tmp += d_a[start+i]*const_mask[i];
        }
    }
    d_result[idx]=tmp;

}

void verify_result(int *array, int *mask, int *result, int n, int m) {
  int radius = m / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < m; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]);
  }
}
int main(){
    int N = 1<<16;
    size_t bytes = N*sizeof(int);
    size_t bytes_m = MASK_SIZE*sizeof(int);

    int * h_a = (int*)malloc(bytes);
    int * h_result = (int*)malloc(bytes);
    int *h_mask = (int*)malloc(bytes_m);

    for(int i = 0; i<N;i++){
        h_a[i] = rand()%100;
    }
    for(int i = 0; i<MASK_SIZE;i++){
        h_mask[i] = rand()%100;
    }

    int*d_a,*d_mask,*d_result;
    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_result,bytes);
    cudaMalloc((void**)&d_mask,bytes_m);

    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,h_mask,bytes_m,cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(const_mask,h_mask,bytes_m);

    int numthreads = 256;
    int numblocks = (N+numthreads - 1)/numthreads;





    conv_1D<<<numblocks,numthreads>>>(d_a,d_result,d_mask,N,MASK_SIZE);
    cudaMemcpy(h_result,d_result,bytes,cudaMemcpyDeviceToHost);
    verify_result(h_a,h_mask, h_result, N, MASK_SIZE);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    conv_1D_constant_memory<<<numblocks,numthreads>>>(d_a,d_result,N);
    cudaMemcpy(h_result,d_result,bytes,cudaMemcpyDeviceToHost);
    verify_result(h_a,h_mask, h_result, N, MASK_SIZE);

    std::cout << "COMPLETED SUCCESSFULLY\n";




    // Free allocated memory on the device and host
    cudaFree(d_result);
    cudaFree(d_mask);
    cudaFree(d_a);





}