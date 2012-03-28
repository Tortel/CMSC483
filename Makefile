all: simplefract.c
	gcc -std=c99 -O2 -lm -lpng -o fractal simplefract.c

# remove emacs backup files
.PHONY: clean
clean:
	rm -f fractal
