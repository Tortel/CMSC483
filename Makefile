all: simplefract.c
	gcc -std=c99 -O2 -lm -lpng -pthread -o cpu-fractal simplefract.c
	# nvcc doesnt need the -pthread flag
	nvcc -O2 -lm -lpng -o gpu-fractal gpu-fractal.cu

# remove emacs backup files
.PHONY: clean
clean:
	rm -f fractal
