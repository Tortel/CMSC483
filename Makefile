# nvcc doesnt need the -pthread flag
all: cpu-fractal.cpp gpu-fractal.cu
	nvcc -O2 -lm -lpng -o gpu-fractal gpu-fractal.cu
	g++ -lm -lpng -pthread -o cpu-fractal cpu-fractal.cpp 

# remove output files
.PHONY: clean
clean:
	rm -f cpu-fractal
	rm -f gpu-fractal
	rm -f *.png
