all:
	nvcc -gencode arch=compute_75,code=sm_75 -o busy_loop busy_loop.cu
debug:
	nvcc -G -gencode arch=compute_75,code=sm_75 -o busy_loop busy_loop.cu
clean:
	rm -rf busy_loop
