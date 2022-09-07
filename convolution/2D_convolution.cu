#include <cassert>
#include <cstdlib>
#include <iostream>
#define MASK_DIM 7
#define MASK_OFFSET (MASK_DIM/2)
__constant__ int mask[MASK_DIM*MASK_DIM];



__global__ void convolution_2d(int *d_matrix,int* d_result,int N){
    int row = threadIdx.x + blockDim.x*blockIdx.x;
    int col = threadIdx.y + blockDim.y *blockIdx.y;

    int startR = row - MASK_OFFSET;
    int startC = col - MASK_OFFSET;

    int temp = 0;
    for(int i = 0 ;i<MASK_DIM;i++){
        for(int j = 0;j<MASK_DIM;j++){
            if(( (startR+i)>=0  &&  (startR+i)<N )  &&  ( (startC+j)>=0  &&  (startC+j)<N )){
                temp+=d_matrix[((startR+i)*N+startC+j)]*mask[i*MASK_DIM+j];
            }
        }
    }
    d_result[row*N+col] = temp;


}

void init_matrix(int *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}

// Verifies the 2D convolution result on the CPU
// Takes:
//  m:      Original matrix
//  mask:   Convolutional mask
//  result: Result from the GPU
//  N:      Dimensions of the matrix
void verify_result(int *m, int *mask, int *result, int N) {
  // Temp value for accumulating results
  int temp;

  // Intermediate value for more readable code
  int offset_r;
  int offset_c;

  // Go over each row
  for (int i = 0; i < N; i++) {
    // Go over each column
    for (int j = 0; j < N; j++) {
      // Reset the temp variable
      temp = 0;

      // Go over each mask row
      for (int k = 0; k < MASK_DIM; k++) {
        // Update offset value for row
        offset_r = i - MASK_OFFSET + k;

        // Go over each mask column
        for (int l = 0; l < MASK_DIM; l++) {
          // Update offset value for column
          offset_c = j - MASK_OFFSET + l;

          // Range checks if we are hanging off the matrix
          if (offset_r >= 0 && offset_r < N) {
            if (offset_c >= 0 && offset_c < N) {
              // Accumulate partial results
              temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
            }
          }
        }
      }
      // Fail if the results don't match
      assert(result[i * N + j] == temp);
    }
  }
}
int  main(){
    int N = 1<<10;
    size_t bytes =N*N *sizeof(int);
    int *h_matrix = new int[N*N];
    int *h_result = new int [N*N];

    init_matrix(h_matrix,N);


    size_t bytes_m = MASK_DIM*MASK_DIM*sizeof(int);

    int *h_mask = new int[MASK_DIM*MASK_DIM];
    init_matrix(h_mask,MASK_DIM);
    int *d_matrix,*d_result;
    cudaMalloc(&d_matrix,bytes);
    cudaMalloc(&d_result,bytes);

    cudaMemcpy(d_matrix,h_matrix,bytes,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask,h_mask,bytes_m);

    int threads = 16;
    int blocks = (N +threads -1)/threads;

    dim3 block_dim(threads,threads);
    dim3 grid_dim(blocks,blocks);
    convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N);


    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);


    verify_result(h_matrix, h_mask, h_result, N);

    std::cout << "COMPLETED SUCCESSFULLY!";

    delete[] h_matrix;
    delete[] h_result;
    delete[] h_mask;

    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;


}