#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Function that split into right array and half array and keeping calling until there is 1 element in the file
__global__ void merge(long* arraylist, long* temparray, int left, int right)
{
    //variable
    int mid = (left + right)/2;
    int length = right - left;
    //stream for right array and left array
    cudaStream_t startleft, startright;
    
    //start to return if the array is 1
    if(length < 2){
        return;
    }
    //create block for the left array
    cudaStreamCreateWithFlags(&startleft, cudaStreamNonBlocking);
    merge<<<1,1,0, startleft>>>(arraylist, temparray, left, mid);
    cudaStreamDestroy(startleft);
    
    //create block for the right array
    cudaStreamCreateWithFlags(&startright, cudaStreamNonBlocking);
    merge<<<1,1,0, startright>>>(arraylist, temparray, mid, right);
    cudaStreamDestroy(startright);

    cudaDeviceSynchronize();
    int i;
    int templeft = left;
    int tempmid = mid;
    //for loop to sort the array
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
    //for loop to copy the array from temparray to pass value
    for(i = left; i < right; i++){
        arraylist[i] = temparray[i];
    }
}
//Function that call the merge function
extern "C" void sort(long ** array, int num){
    long* gpuarray;
    long* temparray;
    int leftnum = 0;
    int rightnum = num;

    int arraysize = num * sizeof(long);
    //allocating memory for the array
    cudaMalloc((void**)&gpuarray, arraysize);
    cudaMalloc((void**)&temparray, arraysize);
    cudaMemcpy(gpuarray, *array, arraysize, cudaMemcpyHostToDevice);
    //call the merge function to start sorting
    merge<<<1, 1>>>(gpuarray, temparray, leftnum, rightnum);
    cudaDeviceSynchronize();
    //copy the array in the sorting function back to the main array
        //free the array    
    cudaMemcpy(*array, gpuarray, arraysize, cudaMemcpyDeviceToHost);
    cudaFree(temparray);
    cudaFree(gpuarray);
    cudaDeviceReset();
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