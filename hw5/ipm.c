#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define FILENAME "random.bin"

int main(int argc, char** argv) {
    int rank;
    int size;
    int bufsize, nrchar;
    long *array;          /* Buffer for reading */
    MPI_Offset filesize;
    MPI_File myfile;    /* Shared file */
    MPI_Status status;  /* Status returned from read */

    MPI_Init(&argc,&argv); // initialize arguments
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // set ranks and size
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    initialize_CUDA(rank);

    MPI_File_open (MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY,
		 MPI_INFO_NULL, &myfile);
    /* Get the size of the file */
    MPI_File_get_size(myfile, &filesize);
    /* Calculate how many elements that is */
    filesize = filesize/sizeof(long);
    /* Calculate how many elements each processor gets */
    bufsize = filesize/size;
    /* Allocate the buffer to read to, one extra for terminating null char */
    array = (long *) malloc((bufsize+1)*sizeof(long));
    /* Set the file view */
    MPI_File_set_view(myfile, rank*bufsize*sizeof(char), MPI_LONG, MPI_LONG, 
                "native", MPI_INFO_NULL);
    /* Read from the file */
    MPI_File_read(myfile, array, bufsize, MPI_LONG, &status);
    /* Find out how many elemyidnts were read */
    MPI_Get_count(&status, MPI_LONG, &nrchar);
    printf("Process %2d read %d characters: ", rank, nrchar);
    MPI_Finalize();
    return 0;
}