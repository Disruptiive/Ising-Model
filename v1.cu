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

void initializeLattice(int* grid, int n,float limit){
    float a = 1.0;
    int i;
    for(i=0;i<n*n;i++){
        grid[i] = ( ((float)rand()/(float)(RAND_MAX)) * a > limit) ? 1 :-1;
    }
}
//calculate periodic boundary conditions
__device__
int getIndex(int pos,int step,int size){
    return (( size + ((pos+step) % size)) % size);
}

__global__
void findSign(int *read,int*write,int size){
    int sum,col_min,col_plus,row_min,row_plus,row,col,idx;
    idx = blockIdx.x * blockDim.x + threadIdx.x; //every thread calculates its "real" index that matches one spin of the grid
    row = idx / size;
    col = idx - row*size;
    sum = 0;
    //find neighbour's (i,j) indexes with respect to periodic boundary conditions and sum them up
    col_min =  getIndex(col,-1,size);
    col_plus = getIndex(col,1,size);
    row_min =  getIndex(row,-1,size);
    row_plus = getIndex(row,1,size);
    sum += read[row*size + col];
    sum += read[row_min*size + col];
    sum += read[row_plus*size+ col];
    sum += read[row*size + col_min];
    sum += read[row*size + col_plus];   
    write[idx] = sum/abs(sum); 
}

void Ising(int *read, int *write, int size){
    int blocksize,numBlocks;
    blocksize = 32;
    numBlocks = (size*size+blocksize-1)/blocksize; //calculate number of blocks based on blocksize so total number of threads == amount of spins in the grid
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
    n = 6400; //n*n grid
    k = 15; //number of iterations
    int *read_arr, *write_arr;
    cudaMallocManaged(&read_arr,  n * n * sizeof(int));
    cudaMallocManaged(&write_arr, n * n * sizeof(int));
    processIsing(read_arr,write_arr,n,k);
    cudaFree(read_arr);
    cudaFree(write_arr);
}