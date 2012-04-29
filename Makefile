# nvcc doesnt need the -pthread flag
all: simplefract.c gpu-fractal.cu
	gcc -std=c99 -O2 -lm -lpng -pthread -o cpu-fractal simplefract.c
	nvcc -arch=sm_20 -O2 -lm -lpng -o gpu-fractal gpu-fractal.cu
	g++ -lm -lpng -pthread -o cpu-julia fracttest.cpp 

# remove output files
.PHONY: clean
clean:
	rm -f cpu-fractal
	rm -f gpu-fractal
	rm -f *.png
