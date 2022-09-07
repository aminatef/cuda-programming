#include<iostream>
#include<vector>
#include<algorithm>
#include<cassert>
#include<numeric>
#include<cstdlib>

using std :: generate;
using std :: vector;
using std :: accumulate;
using std :: cout;
using std :: endl;
#define SHAREDMEM_SIZE 256
__global__ void sumReduction(int * v,int *V_r){
    __shared__ int partial_sum[SHAREDMEM_SIZE];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[idx];
    __syncthreads();
    for(int s = 1;s<blockDim.x;s*=2){
        if((threadIdx.x % (s*2)) == 0 ){
            partial_sum[threadIdx.x] +=partial_sum[threadIdx.x+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        V_r[blockIdx.x] = partial_sum[0];
    }
}

__global__ void sumReduction_with_no_modulo_opt(int * v,int *V_r){
    __shared__ int partial_sum[SHAREDMEM_SIZE];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[idx];
    __syncthreads();
    for(int s = 1;s<blockDim.x;s*=2){
        int index = 2 * s *threadIdx.x;
        if(index < blockDim.x){
            partial_sum[index] +=partial_sum[index+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        V_r[blockIdx.x] = partial_sum[0];
    }

}

__global__ void sumReduction_with_no_bank_conflicts(int * v,int *V_r){
    /*
    The shared memory that can be accessed in parallel is divided into modules
    (also called banks). If two memory locations (addresses)
    occur in the same bank, then you get a bank conflict during which the access is
    done serially, losing the advantages of parallel access.
    */
    
    __shared__ int partial_sum[SHAREDMEM_SIZE];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[idx];
    __syncthreads();
    for(int s = blockDim.x/2 ;s>0;s>>=1){
        if(threadIdx.x < s){
            partial_sum[threadIdx.x] +=partial_sum[threadIdx.x+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        V_r[blockIdx.x] = partial_sum[0];
    }

}

__global__ void sumReduction_half_num_of_threads(int * v,int *V_r){
    __shared__ int partial_sum[SHAREDMEM_SIZE];
    int idx = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[idx]+v[idx+blockDim.x];
    __syncthreads();
    for(unsigned int s = blockDim.x/2 ;s>0;s>>=1){
        if(threadIdx.x < s){
            partial_sum[threadIdx.x] +=partial_sum[threadIdx.x+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        V_r[blockIdx.x] = partial_sum[0];
    }
}
__device__ void warpReduce(volatile int * partial_sum , int tid){
    partial_sum[tid] += partial_sum[tid+32];
    partial_sum[tid] += partial_sum[tid+16];
    partial_sum[tid] += partial_sum[tid+8];
    partial_sum[tid] += partial_sum[tid+4];
    partial_sum[tid] += partial_sum[tid+2];
    partial_sum[tid] += partial_sum[tid+1];
}
__global__ void sumReduction_half_num_of_threads_warp_unrolling(int * v,int *V_r){
    __shared__ int partial_sum[SHAREDMEM_SIZE];
    int idx = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[idx]+v[idx+blockDim.x];
    __syncthreads();
    // warp size = 32 thread
    for(unsigned int s = blockDim.x/2 ;s>32;s>>=1){
        if(threadIdx.x < s){
            partial_sum[threadIdx.x] +=partial_sum[threadIdx.x+s];
        }
        __syncthreads();
    }
    if(threadIdx.x < 32 ){
        warpReduce(partial_sum,threadIdx.x);
    }

    if(threadIdx.x == 0){
        V_r[blockIdx.x] = partial_sum[0];
    }
}

__global__ void sumReduction_load_balancing(int * v,int *V_r,int n){
    __shared__ int partial_sum[SHAREDMEM_SIZE];
    int idx = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
    int gridSize = blockDim.x*2*gridDim.x;
    partial_sum[threadIdx.x] = 0 ;
    while(idx<n){
        partial_sum[threadIdx.x] = v[idx]+v[idx+blockDim.x];
        idx+=gridSize;

    }
    
    __syncthreads();
    // warp size = 32 thread
    for(unsigned int s = blockDim.x/2 ;s>32;s>>=1){
        if(threadIdx.x < s){
            partial_sum[threadIdx.x] +=partial_sum[threadIdx.x+s];
        }
        __syncthreads();
    }
    if(threadIdx.x < 32 ){
        warpReduce(partial_sum,threadIdx.x);
    }

    if(threadIdx.x == 0){
        V_r[blockIdx.x] = partial_sum[0];
    }
}





int RandomNumber(){
    return (rand()%10);
}


int main(){
    int N = 1<<16;
    vector<int> h_a(N);
    vector<int> H_Vr(N);

    int bytes = N * sizeof(int);
    generate(h_a.begin(),h_a.end(),RandomNumber);
    int *V,*V_reduced;

    cudaMalloc(&V,bytes);
    cudaMalloc(&V_reduced,bytes);

    cudaMemcpy(V,h_a.data(),bytes,cudaMemcpyHostToDevice);

    const int TB_SIZE = 256;

    int GRID_SIZE = N / TB_SIZE;




    sumReduction<<<GRID_SIZE,TB_SIZE>>>(V,V_reduced);
    
    sumReduction<<<1,TB_SIZE>>>(V_reduced,V_reduced);

    cudaMemcpy(H_Vr.data(),V_reduced,bytes,cudaMemcpyDeviceToHost);
    
    assert(H_Vr[0] == std::accumulate(h_a.begin(),h_a.end(),0));
    

    cout<<"COMPLETED SUCCESSFULLY"<<endl;



    sumReduction_with_no_modulo_opt<<<GRID_SIZE,TB_SIZE>>>(V,V_reduced);

    sumReduction_with_no_modulo_opt<<<1,TB_SIZE>>>(V_reduced,V_reduced);

    cudaMemcpy(H_Vr.data(),V_reduced,bytes,cudaMemcpyDeviceToHost);
    assert(H_Vr[0] == std::accumulate(h_a.begin(),h_a.end(),0));

    cout<<"COMPLETED SUCCESSFULLY"<<endl;


    sumReduction_with_no_bank_conflicts<<<GRID_SIZE,TB_SIZE>>>(V,V_reduced);

    sumReduction_with_no_bank_conflicts<<<1,TB_SIZE>>>(V_reduced,V_reduced);

    cudaMemcpy(H_Vr.data(),V_reduced,bytes,cudaMemcpyDeviceToHost);
    
    assert(H_Vr[0] == std::accumulate(h_a.begin(),h_a.end(),0));

    cout<<"COMPLETED SUCCESSFULLY"<<endl;

    sumReduction_half_num_of_threads<<<GRID_SIZE,TB_SIZE/2>>>(V,V_reduced);

    sumReduction_half_num_of_threads<<<1,TB_SIZE/2>>>(V_reduced,V_reduced);

    cudaMemcpy(H_Vr.data(),V_reduced,bytes,cudaMemcpyDeviceToHost);
    
    assert(H_Vr[0] == std::accumulate(h_a.begin(),h_a.end(),0));

    cout<<"COMPLETED SUCCESSFULLY"<<endl;




    sumReduction_half_num_of_threads_warp_unrolling<<<GRID_SIZE,TB_SIZE/2>>>(V,V_reduced);

    sumReduction_half_num_of_threads_warp_unrolling<<<1,TB_SIZE/2>>>(V_reduced,V_reduced);
    

    cudaMemcpy(H_Vr.data(),V_reduced,bytes,cudaMemcpyDeviceToHost);
    assert(H_Vr[0] == std::accumulate(h_a.begin(),h_a.end(),0));

    cout<<"COMPLETED SUCCESSFULLY"<<endl;

    sumReduction_load_balancing<<<GRID_SIZE,TB_SIZE/2>>>(V,V_reduced,N);

    sumReduction_load_balancing<<<1,TB_SIZE/2>>>(V_reduced,V_reduced,TB_SIZE);
    

    cudaMemcpy(H_Vr.data(),V_reduced,bytes,cudaMemcpyDeviceToHost);
    assert(H_Vr[0] == std::accumulate(h_a.begin(),h_a.end(),0));

    cout<<"COMPLETED SUCCESSFULLY"<<endl;


    return 0;






    


}