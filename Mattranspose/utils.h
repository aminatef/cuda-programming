#include<assert.h>
#include<stdlib.h>
void verfy_result(float* Matrix_A,float* Matrix_A_transpose,int nx,int ny){
    for(int i = 0;i<ny;i++){
        for(int j = 0;j<nx;j++){
            assert((Matrix_A_transpose[i*nx+j] - Matrix_A[j*nx+i])<0.01);
        }
    }
}