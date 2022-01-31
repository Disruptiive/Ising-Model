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

void initializeLattice(int* grid, int n,float limit){
    float a = 1.0;
    int i;
    for(i=0;i<n*n;i++){
        grid[i] = ( ((float)rand()/(float)(RAND_MAX)) * a > limit) ? 1 :-1;
    }
}

__global__
void findSign(int *read,int*write,int size){
    int sum,col_min,col_plus,row_min,row_plus,row,col,i,j,row_col_workload,start_row,end_row,start_col,end_col;
    //calculate how many and which rows and columns from the grid the thread must compute
    row_col_workload = size/gridDim.x;
    start_row = (blockIdx.x*size)/gridDim.x;
    end_row = start_row + row_col_workload;
    start_col = (blockIdx.y*size)/gridDim.y;
    end_col = start_col + row_col_workload;
    for (i=start_row;i<end_row;i+=1){
        for(j=start_col;j<end_col;j++){
            row = i;
            col = j;
            sum = 0;
            col_min = getIndex(col,-1,size);
            col_plus = getIndex(col,1,size);
            row_min = getIndex(row,-1,size);
            row_plus = getIndex(row,1,size);
            sum += read[row*size + col];
            sum += read[row_min*size + col];
            sum += read[row_plus*size+ col];
            sum += read[row*size + col_min];
            sum += read[row*size + col_plus];   
            write[row*size + col] = sum/abs(sum);
        }
    }
}
void Ising(int *read, int *write, int size){
    int b = 64;
    dim3 blocksize(1,1); //1 thread per block
    dim3 numBlocks(size/b,size/b); //calculate amount of blocks based on grid size 
    findSign<<<numBlocks,blocksize>>>(read,write,size);  
    cudaDeviceSynchronize();
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
    float limit = 0.5;
    initializeLattice(read,size,limit);
    for(i=0;i<iterations;i++){
        //writeMatrix(read,i,size);
        Ising(read, write, size);
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