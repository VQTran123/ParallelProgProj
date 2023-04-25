#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clockcycle.h"
#include <mpi.h>

#define ARRAY_SIZE 1048576
#define clock_frequency 512000000
#define FILENAME "random.bin"

void initialize_CUDA(int rank);
void sort(long **g_idata, int n);

// serial merge process of merge sort algorithm
long * merge(long* arr, int l, int m, int r)
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
    free(L);
    free(R);
    return arr;
}

// serial mergesort driver
long * mergeSort(long* arr, int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
 
        arr = mergeSort(arr, l, m);
        arr = mergeSort(arr, m + 1, r);
 
        arr = merge(arr, l, m, r);
    }
    return arr;
}

// sends elements from one rank to another
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
                i++;
            }
            sendbuf = merge(sendbuf, 0, arr_size, arr_size*2);
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

    // parallel i/o file reading
    MPI_File fh;
    MPI_Offset filesize, valuesize, offset;
    long* buffer;
    long num_values, values_per_process;

    MPI_File_open(MPI_COMM_WORLD, "random.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_File_get_size(fh, &filesize);
    valuesize = sizeof(long);
    num_values = filesize / valuesize;
    values_per_process = num_values / size;

    // set view and read a million long values from binary file
    offset = rank * values_per_process * valuesize;
    MPI_File_set_view(fh, offset, MPI_LONG, MPI_LONG, "native", MPI_INFO_NULL);

    buffer = (long*) malloc(values_per_process * sizeof(long));
    MPI_File_read_all(fh, buffer, values_per_process, MPI_LONG, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    int elements = ARRAY_SIZE/size;
    
    /*
    long *array = (long*)malloc(elements*sizeof(long));
    for(int i = 0; i < elements; i++){
        array[i] = (long) rank*elements+i;
    }
    */

    double start_cycles;
    double end_cycles;
    double reduceTime;
    int step = 1;
    long * placeholder_array;

    if(rank == 0) {
        start_cycles=clock_now();
    }

    // call the cuda merge sort for each processes' chunk
    sort(&buffer, elements);

    // continual merging followed by sorting until rank 0 is left with all elements merged and sorted
    while(step < size) {
        placeholder_array = (long*)malloc(elements*step*sizeof(long));
        buffer = rank_merge(buffer, placeholder_array, elements*step, MPI_LONG, MPI_COMM_WORLD, step);
        step *= 2;
    }

    // output the time difference statistics from serial merge and parallel merge
    if(rank == 0){
        end_cycles=clock_now(); 
        reduceTime = (end_cycles-start_cycles)/clock_frequency;

        for(int i = 0; i < ARRAY_SIZE; i++) {
            printf("Value is: %ld", buffer[i]);
        }

        printf("Parallelized process took %f seconds.\n",reduceTime);
        
        start_cycles = clock_now();
        buffer = mergeSort(buffer,0,ARRAY_SIZE-1);
        end_cycles = clock_now();
        
        reduceTime = (end_cycles-start_cycles)/clock_frequency;
        
        printf("Serialized process took %f seconds.\n",reduceTime);
    }
    free(buffer);
    free(placeholder_array);
    MPI_Finalize();
    return 0;
}