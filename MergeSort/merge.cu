#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define block_size 1024

__global__ void merge(int* arraylist, int* temparray, int left, int right, int rank)
{
    //variable
    int mid = (left + right)/2;
    int length = right - left;
    //stream for right array and left array
    cudaStream_t startleft, startright;
    
    //start to return if the array is 1 or 0
    if(length < 2){
        return;
    }
    //create block for the left array
    cudaStreamCreateWithFlags(&startleft, cudaStreamNonBlocking);
    merge<<<1,1,0, startleft>>>(arraylist, temparray, left, mid, rank + 1);
    cudaStreamDestroy(startleft);
    
    //create block for the right array
    cudaStreamCreateWithFlags(&startright, cudaStreamNonBlocking);
    merge<<<1,1,0,startright>>>>(arraylist, temparray, mid, right, rank + 1);
    cudaStreamDestroy(startright);

    cudaDeviceSynchronize();
    int i;
    int templeft = left;
    int tempmid = mid;
    for(i = left; i < right; i++){
        if(templeft < mid && (tempmid >= right || arraylist[templeft] <= arraylist[tempmid])){
            temparray[i] = arraylist[templeft];
            templeft++;
        }
        else{
            temparray[i] = arraylist[tempmid];
            tempmid++;
        }
    }
    for(i = left; i < right; index++){
        arraylist[i] = temparray[i];
    }
}

extern "C" void sort(long *g_idata, long *g_odata, int n){
    
}

extern "C" void initialize_CUDA(int rank){
    int cudaDeviceCount;
    int cE = cudaGetDeviceCount( &cudaDeviceCount);
    if( cE != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n",
            cE, cudaDeviceCount );
        exit(-1);
    }
    cE = cudaGetDeviceCount( &cudaDeviceCount);
    if( cE != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
            rank, (rank % cudaDeviceCount), cE);
        exit(-1);
    }
}