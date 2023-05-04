all: mpi.c merge.cu
	mpixlc -O3 mpi.c -c -o mpi.o
	nvcc -arch=sm_52 -rdc=true -lcudadevrt -O3 merge.cu -c -o cuda.o
	nvcc -arch=sm_52 -dlink -o merge.o cuda.o -L/usr/local/cuda-11.2/lib64/ -lcudart -lcudadevrt
	mpixlc cuda.o -o merge.o mpi.o -L/usr/local/cuda-11.2/lib64/ -lcudart -lcudadevrt