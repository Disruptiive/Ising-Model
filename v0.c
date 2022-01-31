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

//calculate periodic boundary conditions
int getIndex(int pos,int step,int size){
    return (( size + ((pos+step) % size)) % size);
}

void initializeLattice(int* grid, int n){
    float a = 1.0;
    int i,count,row;
    for(i=0;i<n*n;i++){
        grid[i] = ( ((float)rand()/(float)(RAND_MAX)) * a > 0.5) ? 1 :-1;
    }
}

int findSign(int *grid,int col,int row,int size){
    int sum,col_min,col_plus,row_min,row_plus;
    sum = 0;
    //find neighbour's (i,j) indexes with respect to periodic boundary conditions and sum them up
    col_min = getIndex(col,-1,size);
    col_plus = getIndex(col,1,size);
    row_min = getIndex(row,-1,size);
    row_plus = getIndex(row,1,size);
    sum += grid[row*size + col];
    sum += grid[row_min*size + col];
    sum += grid[row_plus*size+ col];
    sum += grid[row*size + col_min];
    sum += grid[row*size + col_plus];
    return sum;
}

void Ising(int *read, int *write, int size){
    int sign,i,count,row,col;
    for(i=0;i<size*size;i++){
        //read and write are 1d arrays that represent matrix so we need to find based on i row and column of every spin
        row = i / size; 
        col = i - row*size;
        sign = findSign(read,col,row,size);
        write[i] = sign/abs(sign); //divide sign by the absolute value of sign to get +1 or -1 value
    }
}

void writeMatrix(int *matrix, int k,int n){
    int i;
    FILE *f;
    char filename[20];
    sprintf(filename,"matrix_step-%d.txt",k);
    f = fopen(filename,"wb");
    fwrite(&n,sizeof(int),1,f);
    fwrite(matrix,sizeof(int),n*n,f);
    fclose(f);
}


void processIsing(int *read,int *write,int size,int iterations){
    int i,j;
    initializeLattice(read,size); //initiate grid from a uniform state of +1 and -1 spins
    for(i=0;i<iterations;i++){
        //writeMatrix(read,i,size); //write current state of grid to file
        Ising(read, write, size); 
        swapptrs(&read,&write); //swap read and write pointers
    }
    //writeMatrix(write,i,size); 

}

int main(){
    int n,k;
    n = 6400; //n*n grid
    k = 15; //number of iterations
    int *read_arr, *write_arr;
    read_arr = malloc(n * n * sizeof(int));
    write_arr = malloc(n * n * sizeof(int));
    processIsing(read_arr,write_arr,n,k);
    free(read_arr);
    free(write_arr);
}