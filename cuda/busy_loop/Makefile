COMPUTE_CAP=75

all:
	nvcc -gencode arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -o busy_loop busy_loop.cu
debug:
	nvcc -G -gencode arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -o busy_loop busy_loop.cu
clean:
	rm -rf busy_loop
