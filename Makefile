all: mpi.c merge.cu
	mpixlc -O3 mpi.c -c -o mpi.o
	nvcc -arch=sm_52 -rdc=true -lcudadevrt -O3 merge.cu -c -o cuda.o
	mpicc -O3 mpi.o cuda.o -o merge.o \
	 -L/usr/local/cuda-11.2/lib64/ -lcuda -lcudart