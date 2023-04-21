#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clockcycle.h"
#include <mpi.h>

#define ARRAY_SIZE 1048576
#define clock_frequency 512000000

void initialize_CUDA(int rank);
void sort(long *g_idata, int n);


void merge(long* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
 
    long * L = calloc(n1,sizeof(long));
    long * R = calloc(n2,sizeof(long));
    
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(long* arr, int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
 
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
 
        merge(arr, l, m, r);
    }
}

long * rank_merge(long *sendbuf, long *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm, int step) {
    int rank;
    MPI_Comm_rank(comm,&rank);
    int size;
    MPI_Comm_size(comm,&size);
    if(rank % step == 0) {
        int pair = rank ^ step;
        if(rank > pair) {
            MPI_Send(sendbuf,count,datatype,pair,1,comm);
        }
        else {
            MPI_Recv(recvbuf, count, datatype, pair, 1, comm, MPI_STATUS_IGNORE);
            int arr_size = *(&recvbuf + 1) - recvbuf;
            sendbuf = realloc(sendbuf, (long) arr_size*2*sizeof(long));
            int i = arr_size;
            while(i < arr_size*2){
                sendbuf[i] = recvbuf[i-arr_size];
            }
            merge(sendbuf, 0, arr_size, arr_size*2);
        }
    }
    return sendbuf;
}

int main(int argc, char** argv) {
    int rank;
    int size;

    MPI_Init(&argc,&argv); // initialize arguments
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // set ranks and size
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    initialize_CUDA(rank);

    //Compute local array sum
    int elements = ARRAY_SIZE/size;
    long *array = (long*)malloc(elements*sizeof(long));
    for(int i = 0; i < elements; i++){
        array[i] = (long) rank*elements+i;
    }

    double start_cycles;
    double end_cycles;
    double reduceTime;
    int step = 1;
    long * placeholder_array;

    if(rank == 0) {
        start_cycles=clock_now();
    }

    sort(array, elements);

    while(step < size) {
        placeholder_array = (long*)malloc(elements*step*sizeof(long));
        array = rank_merge(array, placeholder_array, 0, MPI_LONG, MPI_COMM_WORLD, step);
        step *= 2;
    }

    if(rank == 0){
        end_cycles=clock_now(); 
        reduceTime = (end_cycles-start_cycles)/clock_frequency;

        printf("Parallelized process took %f seconds.\n",reduceTime);
        
        start_cycles = clock_now();
        mergeSort(array,0,ARRAY_SIZE-1);
        end_cycles = clock_now();
        
        reduceTime = (end_cycles-start_cycles)/clock_frequency;
        
        printf("Serialized process took %f seconds.\n",reduceTime);
    }
    MPI_Finalize();
    return 0;
}
