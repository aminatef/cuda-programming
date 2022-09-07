#include<stdio.h>
#include<cuda_runtime.h>
//#include<helper_cuda.h>
/*
    A kernel is defined using the __global__declaration specifier
    and the number of CUDA threads that execute that kernel for a given kernel call is specified using
    <<<...>>>
*/
__global__ void vector_add(const float*A,const float *B,float *C,int num_Elements){
    /* 
    grid is a 1d-2d-3d set of blocks
    block is a set of threads
    blockdim is the size of the block 
    blockidx is the index of the block in the grid
    threadidx is the thread index in the block
    */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<num_Elements){
        C[i] = A[i]+B[i]+0.0f;
    }
}
//HOST CODE
/*
    Linear memory is typically allocated using cudaMalloc()and freed using cudaFree()
    and data transfer between host memory and device memory are typically
    done using cudaMemcpy()
*/
int main(){
    cudaError_t error = cudaSuccess;
    int numElements = 50000;
    size_t size =sizeof(float)*numElements;
    float* host_A = (float*)malloc(size);
    float* host_B = (float*)malloc(size);
    float* host_C = (float*)malloc(size);

    if(host_A==NULL||host_B==NULL||host_C==NULL){
        fprintf(stderr,"Falied to allocate host vectors");
        exit(EXIT_FAILURE);
    }


    for(int i = 0 ;i<numElements;i++){
        host_A[i] = rand()/float(RAND_MAX);
        host_B[i] = rand()/float(RAND_MAX);
    }


    // device memory allocation
    float *device_A = NULL;
    error = cudaMalloc((void **)&device_A,size);
    if (error != cudaSuccess){
        fprintf(stderr,"Falied to allocate device vector A\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float *device_B = NULL;
    error = cudaMalloc((void **)&device_B,size);
    if (error != cudaSuccess){
        fprintf(stderr,"Falied to allocate device vector B\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    float *device_C = NULL;
    error = cudaMalloc((void **)&device_C,size);
    if (error != cudaSuccess){
        fprintf(stderr,"Falied to allocate device vector C\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error  = cudaMemcpy(device_A,host_A,size,cudaMemcpyHostToDevice);
    error  = cudaMemcpy(device_B,host_B,size,cudaMemcpyHostToDevice);
    
    //KERNEL INVOKATION
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements+threadsPerBlock-1)/threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,threadsPerBlock);
    vector_add<<<blocksPerGrid,threadsPerBlock>>>(device_A,device_B,device_C,numElements);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  error = cudaMemcpy(host_C,device_C,size,cudaMemcpyDeviceToHost);
  for (int i = 0; i < numElements; ++i) {
    if (fabs(host_A[i] + host_B[i] - host_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");









}

