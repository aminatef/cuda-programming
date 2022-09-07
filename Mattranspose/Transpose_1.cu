#include<iostream>
#include<stdlib.h>
#include"utils.h"
#define TILE_SIZE 32
#define BLOCK_ROWS 32


__global__ void transposeCoalesced(float * out,float* in,const int nx,int ny){
    __shared__ float tile [TILE_SIZE][TILE_SIZE];
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int width = TILE_SIZE*gridDim.x;

    for(int i = 0 ;i<TILE_SIZE;i+=BLOCK_ROWS){
        tile[(threadIdx.y+i)][threadIdx.x] = in[(y+i)*width+x];
    }
    __syncthreads();
    //transposed offsets
    y = blockDim.x*blockIdx.x + threadIdx.x;
    x = blockDim.y*blockIdx.y + threadIdx.y;
    for(int i=0 ; i<TILE_SIZE;i+=BLOCK_ROWS){
        out[(y+i)*width+x] = tile[threadIdx.y+i][threadIdx.x];
    }
}
int main(){

    int nx = 1<<8 ,ny = 1<<8;

    //kernel and block size 
    int blockX = 32;
    int blockY= 32 ;
    
    size_t bytes= nx*ny*sizeof(float);

    //kernel config
    dim3 block(blockX,blockY);
    dim3 grid((nx+blockX-1)/blockX , (ny+blockY-1)/blockY);

    //allocate host memory
    float * h_A = (float*) malloc(bytes);
    float *gpuRef = (float*)malloc(bytes);

    for(int i = 0;i<nx*ny;i++){
        h_A[i] = rand()/(float)RAND_MAX;
    }
    float *d_A , *d_C;
    cudaMalloc((float**)&d_A,bytes);
    cudaMalloc((float**)&d_C,bytes);
    
    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);

    transposeCoalesced<<<grid,block>>>(d_C,d_A,nx,ny);
    cudaMemcpy(gpuRef,d_C,bytes,cudaMemcpyDeviceToHost);

    verfy_result(h_A,gpuRef,nx,ny);

    std::cout<<"COMPLETED SUCCESFULLY"<<std::endl;

}

