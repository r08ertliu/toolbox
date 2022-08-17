all:
	g++ -o host host.cpp
	nvcc -lcuda -lnvidia-ml -o kernel kernel.cu
clean:
	rm -rf host kernel
