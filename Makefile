all: mpi.c merge.cu
	mpixlc -O3 mpi.c -c -o mpi.o
	nvcc -O3 merge.cu -c -o cuda.o
	mpixlc -O3 mpi.o cuda.o -o merge-exe \
	 -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++