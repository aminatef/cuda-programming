
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::ios;
using std::ofstream;
using std::vector;
__global__ void histogram(int * d_a,int*d_result,int size){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    while(i<size){
        atomicAdd(&(d_result[d_a[i]]),1);
        i+=stride;
    }
}
__global__ void histogram_shared(int * d_a,int*d_result,int size){
    __shared__ int temp[256];
    temp[threadIdx.x]=0;
    __syncthreads();
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    while(i<size){
        atomicAdd(&(temp[d_a[i]]),1);
        i+=stride;
    }
    __syncthreads();
    atomicAdd(&d_result[threadIdx.x],temp[threadIdx.x]);

}
int main(){
    int *h_a;
    long *h_histo;
    int N = 1<<20;
    size_t size =256*sizeof(int);
    h_a = new int[256];
    h_histo = new long[256];
    for (int i = 0;i<256;i++){
      h_a[i] = rand()%256;
    }
    int*dev_a,*dev_histo;
    cudaMalloc((void**)&dev_a,size);
    cudaMalloc((void**)&dev_histo,256*sizeof(long));
    cudaMemset(dev_histo,0,size);

    cudaMemcpy(dev_a,h_a,size,cudaMemcpyDeviceToHost);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    int blocks= prop.multiProcessorCount;
    histogram<<<2*blocks,256>>>(dev_a,dev_histo,N);







}