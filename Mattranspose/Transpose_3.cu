#include<iostream>
#include<stdlib.h>
#include"utils.h"
#define TILE_SIZE 32
#define BLOCK_ROWS 32

__global__ void transposeDiagonal(float * out,float* in,const int width,int height){
    __shared__ float tile [TILE_SIZE][TILE_SIZE+1];

    int block_idx_x,block_idx_y;

    if(width == height){
        block_idx_y = blockIdx.x;
        block_idx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    }else{
        int bid = blockIdx.x+gridDim.x*blockIdx.y;
        block_idx_y = bid%gridDim.y;
        block_idx_x = ((bid/gridDim.y)+block_idx_y)%gridDim.x;
    }
    int xindex = block_idx_x*TILE_SIZE + threadIdx.x;
    int yindex = block_idx_y*TILE_SIZE + threadIdx.y;

    int index_in = xindex +(yindex*width);

    yindex = block_idx_x*TILE_SIZE + threadIdx.x;
    xindex = block_idx_y*TILE_SIZE + threadIdx.y;

    int index_out = xindex +(yindex*width);

    for(int i = 0 ;i<TILE_SIZE;i+=BLOCK_ROWS){
        tile[(threadIdx.y+i)][threadIdx.x] = in[index_in+i*width];
    }
    __syncthreads();

    for(int i =0 ; i<TILE_SIZE;i+=BLOCK_ROWS){
        out[index_out+i*height] = tile[threadIdx.y+i][threadIdx.x];
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

    transposeDiagonal<<<grid,block>>>(d_C,d_A,nx,ny);
    cudaMemcpy(gpuRef,d_C,bytes,cudaMemcpyDeviceToHost);

    verfy_result(h_A,gpuRef,nx,ny);

    std::cout<<"COMPLETED SUCCESFULLY"<<std::endl;

}
