#include<iostream>
#include<stdlib.h>
#include<assert.h>
#include"utils.h"
#define TILE_SIZE 32
#define BLOCK_ROWS 32

__global__ void transposeNaiveRow(float * out,float* in,const int nx,int ny){
    int x = threadIdx.x + blockDim.x *blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    out[y + ny*x] = in[x + nx*y];
}
__global__ void transposeNaiveCols(float * out,float* in,const int nx,int ny){
    int x = threadIdx.x + blockDim.x *blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    out[x + nx*y] = in[y + ny*x];
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

    transposeNaiveCols<<<grid,block>>>(d_C,d_A,nx,ny);
    cudaMemcpy(gpuRef,d_C,bytes,cudaMemcpyDeviceToHost);

    verfy_result(h_A,gpuRef,nx,ny);

    std::cout<<"COMPLETED SUCCESFULLY"<<std::endl;

}
