#include<stdio.h>
#include<cassert>
#include<iostream>

using std :: cout;

__global__ void vectorAddUm(int *a,int*b,int *c,int N){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id<N){
        c[id] = a[id]+b[id];
    }
}

int main(){
    int N = 50000;
    size_t size = sizeof(float) * N;
    int * a,*b,*c;
    cudaMallocManaged(&a,size);
    cudaMallocManaged(&b,size);
    cudaMallocManaged(&c,size);


    // device id 

    int id  = cudaGetDevice(&id);
    /*
    cudaMallocManaged allows for oversubscription,
    and with the correct cudaMemAdvise policies enabled,
    will allow the application to retain most if not all the performance of cudaMalloc.
    cudaMallocManaged also won't force an allocation to be resident until it is needed
    or prefetched, reducing the overall pressure on the operating system schedulers and
    better enabling multi-tenet use cases
    */
   cudaMemAdvise(a,size,cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId);
   cudaMemAdvise(b,size,cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId);

   cudaMemPrefetchAsync(c,size,id);

   for(int i=0;i<N;i++){
    a[i] = rand()%100;
    b[i] = rand()%100;
   }
   cudaMemAdvise(a,size,cudaMemAdviseSetReadMostly,id);
   cudaMemAdvise(b,size,cudaMemAdviseSetReadMostly,id);
   cudaMemPrefetchAsync(a,size,id);
   cudaMemPrefetchAsync(b,size,id);
   int numThreads = 1024;
   int gridSize = (N + numThreads-1)/numThreads;
   vectorAddUm<<<gridSize , numThreads>>>(a,b,c,N);
   // We need this because we don't get the implicit synchronization of
   // cudaMemcpy like in the original example
   cudaDeviceSynchronize();
   cudaMemPrefetchAsync(a,size,cudaCpuDeviceId);
   cudaMemPrefetchAsync(b,size,cudaCpuDeviceId);
   cudaMemPrefetchAsync(c,size,cudaCpuDeviceId);

   for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }

  // Free unified memory (same as memory allocated with cudaMalloc)
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cout << "COMPLETED SUCCESSFULLY!\n";

  return 0;




}