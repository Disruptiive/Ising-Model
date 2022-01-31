#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h> 
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <limits.h>

void swapptrs(int **a,int **b){
    int *tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__
int getIndex(int pos,int step,int size){
    return (( size + ((pos+step) % size)) % size);
}

void initializeLattice(int* grid, int n){
    float a = 1.0;
    int i;
    for(i=0;i<n*n;i++){
        grid[i] = ( ((float)rand()/(float)(RAND_MAX)) * a > 0.5) ? 1 :-1;
    }
}

__global__
void findSign(int *read,int*write,int size){
    int sum,row,col,i,j,row_col_workload,start_row,start_col,buff_start_row,buff_start_col;
    extern __shared__ int sharedBuf[];
    //find the submatrix of the grid that the threads have to compute
    row_col_workload = size/gridDim.x;
    start_row = (blockIdx.x*size)/gridDim.x;
    start_col = (blockIdx.y*size)/gridDim.y;
    //we include 2 more rows and columns in the shared memory buffer so that every neighbour is included in the shared buffer
    buff_start_row =  getIndex(start_row,-1,size);
    buff_start_col =  getIndex(start_col,-1,size);
    //fill the shared memory with values from read_arr
    for(j=threadIdx.y; j<row_col_workload+2; j+=blockDim.y){
        for(i=threadIdx.x; i<row_col_workload+2; i+=blockDim.x){
            row = getIndex(buff_start_row,j,size);
            col = getIndex(buff_start_col,i,size);
            sharedBuf[j * (row_col_workload+2) + i] = read[row * size + col];
        }
    }
    __syncthreads();
    //calculate the sign for every value in the b*b sumatrix
    for(j=threadIdx.y+1; j<row_col_workload+1; j+=blockDim.y){
        for(i=threadIdx.x+1; i<row_col_workload+1; i+=blockDim.x){
            sum = 0;
            row = getIndex(buff_start_row,j,size);
            col = getIndex(buff_start_col,i,size);
            sum += sharedBuf[j * (row_col_workload+2) + i];
            sum += sharedBuf[j * (row_col_workload+2) + i+1];
            sum += sharedBuf[j * (row_col_workload+2) + i-1];
            sum += sharedBuf[(j+1) * (row_col_workload+2) + i];
            sum += sharedBuf[(j-1) * (row_col_workload+2) + i];
            
            write[row * size + col] = sum/abs(sum);
        }
    }
    
}
void writeMatrix(int *matrix, int k,int n){
    FILE *f;
    char filename[20];
    sprintf(filename,"matrix_step-%d.txt",k);
    f = fopen(filename,"wb");
    fwrite(&n,sizeof(int),1,f);
    fwrite(matrix,sizeof(int),n*n,f);
    fclose(f);
}


void processIsing(int *read,int *write,int size,int iterations){
    int i;
    initializeLattice(read,size);
    int threads_per_block = 32;
    int b = 64;
    dim3 blocksize(threads_per_block,threads_per_block); //(32,32) threads per block
    dim3 numBlocks(size/b,size/b); //number of blocks is calculated based on the size of the grid

    for(i=0;i<iterations;i++){
        //writeMatrix(read,i,size);
        findSign<<<numBlocks,blocksize,(b+2)*(b+2)*sizeof(int)>>>(read,write,size);  
        cudaDeviceSynchronize();

        swapptrs(&read,&write);
    }
    //writeMatrix(write,i,size);

}

int main(){
    int n,k;
    n = 6400; 
    k = 15;
    int *read_arr, *write_arr;
    cudaMallocManaged(&read_arr,  n * n * sizeof(int));
    cudaMallocManaged(&write_arr, n * n * sizeof(int));
    processIsing(read_arr,write_arr,n,k);
    cudaFree(read_arr);
    cudaFree(write_arr);
}