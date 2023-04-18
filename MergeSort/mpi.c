#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clockcycle.h"
#include <mpi.h>

#define ARRAY_SIZE 1048576
#define clock_frequency 512000000

void initialize_CUDA(int rank);
void sort(long *g_idata, long *g_odata, int n);

int rank_merge(long *sendbuf, long *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm, int step, long * right_arr) {
    int rank;
    MPI_Comm_rank(comm,&rank);
    int size;
    MPI_Comm_size(comm,&size);
    if(rank % step == 0) {
        int pair = rank ^ step;
        if(rank > pair) {
            MPI_Send(&sendbuf,count,datatype,pair,1,comm);
        }
        else {
            MPI_Recv(&recvbuf, count, datatype, pair, 1, MPI_STATUS_IGNORE);
            *right_arr = *recvbuf;
        }
    }
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
        array[i] = rank*elements+i;
    }

    long *finalArray = (*long)malloc(ARRAY_SIZE*sizeof(long));
    long *right_array = (long*)malloc(elements*sizeof(long));

    /*
    double start_cycles
    if(rank == 0) {
        start_cycles = clock_now();
    }
    sort(&array,&finalArray,ARRAY_SIZE);
    */


    if(rank == 0){
        rank_merge(&array, &finalArray, 0, MPI_LONG, MPI_COMM_WORLD, 2, &right_array);
        double start_cycles=clock_now();
        sort(&array,&finalArray,ARRAY_SIZE);
        double end_cycles=clock_now(); 

        double reduceTime = (end_cycles-start_cycles)/clock_frequency;

        printf("Parallelized process took %d seconds.\n",reduceTime);

        start_cycles = clock_now();
        mergeSort(array,0,ARRAY_SIZE-1);
        end_cycles = clock_now();
        
        reduceTime = (end_cycles-start_cycles)/clock_frequency;
        
        printf("Serialized process took %d seconds.\n",reduceTime);
    }

    else{
        rank_merge(&array, &finalArray, 0, MPI_LONG, MPI_COMM_WORLD, 2, &right_array);
        sort(&finalArray,&finalArray,ARRAY_SIZE);
    }
    MPI_Finalize();
    return 0;
}

void merge(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
 
    int L[n1], R[n2];
 
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

void mergeSort(int arr[], int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
 
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
 
        merge(arr, l, m, r);
    }
}