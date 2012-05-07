/*
Sample taken from here:
http://www.labbookpages.co.uk/software/imgProc/libPNG.html
*/

#include <stdio.h>
//#include <math.h>
#include <malloc.h>
#include <png.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <cuda.h>
#include "error.h"

//Number of images/threads
#define NUM 9

//Debugging printing
#define WRITE_V 0
#define J_SCALE 1.5f

//Structure for passing data to the threads
typedef struct _thread_data_t{
	int tid;
	char *filename;
	int size;
	float *buffer;
} thread_data_t;

//Device constants
//Original function call:
// float *createMandelbrotImage(int size, float xS, float yS, float rad, int maxIteration);
__constant__ float xS;
__constant__ float yS;
__constant__ float rad;
__constant__ int devMaxIteration;
__constant__ int devSize;


// Creates a test image for saving. Creates a Mandelbrot Set fractal of size size x size
__global__ void createMandelbrotImage(float *devBuffer);

//Julia kernel code
__global__ void j_kernel(float *ptr );
__device__ float julia(int x, int y );

// This takes the float value 'val', converts it to red, green & blue values, then
// sets those values into the image memory buffer location pointed to by 'ptr'
inline void setRGB(png_byte *ptr, float val);

// This function actually writes out the PNG image file
void *writeImage( void *input);

//Functions to run the specific code
void doMandelbrot(int size);
void doJulia(int size);


/**
 * Julia complex number struct
 */
struct cuComplex {
   float   r;
   float   i;
   __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
   __device__ float magnitude2( void ) { return r * r + i * i; }
   __device__ cuComplex operator*(const cuComplex& a) {
      return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
   }
   __device__ cuComplex operator+(const cuComplex& a) {
      return cuComplex(r+a.r, i+a.i);
   }
};

int main(int argc, char *argv[])
{
	if(argc != 3){
		printf("Usage: gpu-fractal <Image size> <m or j>\n");
		printf("\t*Image size is an integer, and is the size of the square image to output\n");
		printf("\t*Use m/M to generate a Mandelbrot fractal, j/J for a Julia fractal\n");
		return -1;
	}
   //Image size as first parameter
   int size;
   size = atoi(argv[1]);

   int mandelbrot = 0;

   //Julia or Mandelbrot image
   if(argv[2][0] == 'm' || argv[2][0] == 'M')
	   mandelbrot = 1;

   //Timers
   struct timeval start, end;

   gettimeofday(&start, NULL);

   if(mandelbrot){
	   doMandelbrot(size);
   } else {
	   doJulia(size);
   }

   gettimeofday(&end, NULL);

   printf("\nTotal time: %lf\n", (end.tv_sec + (end.tv_usec/1000000.0)) - (start.tv_sec + (start.tv_usec/1000000.0)));

   printf("Done!\n");
   return 0;
}

/**
 * Handles the whole mandelbrot process
 */
void doMandelbrot(int size){
	//Timers
	struct timeval start, end;
	//Number of iterations per image
	int iterations[] = {20, 25, 30, 35, 40, 50, 100, 300, 500, 1000};

	//Array to keep track of pthreads
	pthread_t threads[NUM];

	char out[] = "Mandelbrot-0.png";
	int outSize = strlen(out) + 1;

	//Copy size to the constant memory
	HANDLE_ERROR( cudaMemcpyToSymbol(devSize, &size, sizeof(int)) );

	//Allocate the device image buffer
	float *devBuffer;
	HANDLE_ERROR( cudaMalloc( (void **) &devBuffer, size * size * sizeof(float)) );

	//Number of cuda threads to start
	dim3  grid(size, size);

	//Temp var for parameters
	float *tmp;
	tmp = (float *) malloc(sizeof(float));

	for(int pos = 0; pos < NUM; pos++){
		printf("Iteration %i\n", pos);

		//11 is the position of the number in the filename
		out[11] = (char) '0' + pos;

		float minMu = iterations[pos];
		float maxMu = 0;

		// Create a test image - in this case a Mandelbrot Set fractal
		// The output is a 1D array of floats, length: size * size
		//float *buffer = createMandelbrotImage(size, -0.802, -0.177, 0.011, 100);

		//Start Timer
		gettimeofday(&start, NULL);

		//Copy parameters into device constant memory
		//Need to use a temp variable to hold the parameters (Might be changed to program arguments)
		*tmp = -0.802f;
		HANDLE_ERROR( cudaMemcpyToSymbol(xS, tmp, sizeof(float)) );
		*tmp = -0.177f;
		HANDLE_ERROR( cudaMemcpyToSymbol(yS, tmp, sizeof(float)) );
		*tmp = 0.011f;
		HANDLE_ERROR( cudaMemcpyToSymbol(rad, tmp, sizeof(float)) );
		//Max iteration is in an array
		HANDLE_ERROR( cudaMemcpyToSymbol(devMaxIteration, &(iterations[pos]), sizeof(int)) );

		//Clear device memory
		HANDLE_ERROR( cudaMemset(devBuffer, 0, size * size * sizeof(float)) );


		//Allocate the memory for the image
		float *buffer;
		buffer = (float *) malloc(size * size * sizeof(float));

		//Start the kernel
		createMandelbrotImage<<<grid,1>>>(devBuffer);

		//Copy from device
		HANDLE_ERROR( cudaMemcpy( buffer,  devBuffer, size*size*sizeof(float), cudaMemcpyDeviceToHost ) );

		//Find mu min/max
		for(int i = 0; i < size * size; i ++){
			if(buffer[i] < minMu) minMu = buffer[i];
			if(buffer[i] > maxMu) maxMu = buffer[i];
		}

		//Normalize buffer values
		for(int i = 0; i < size * size; i++){
			buffer[i] = (buffer[i] - minMu) / (maxMu - minMu);
		}

		//End timer
		gettimeofday(&end, NULL);

		printf("Iteration %i time: %lf\n", pos, (end.tv_sec + (end.tv_usec/1000000.0)) - (start.tv_sec + (start.tv_usec/1000000.0)));

		// Save the image to a PNG file
		//Set up the struct
		thread_data_t data;
		data.buffer = buffer;
		//Use the iteration number as TID
		data.tid = pos;
		data.size = size;

		//Have to copy the filename
		data.filename = (char *) malloc( outSize * sizeof(char) );
		strncpy(data.filename, out, outSize);
		//Always make sure to terminate
		data.filename[outSize - 1] = '\0';

		printf("Saving PNG\n\n");
		//Start the pthread
		pthread_create(&(threads[pos]), NULL, writeImage, (void *) &data);

		//pthread_join(threads[pos], NULL);

		//writeImage( (void *) &data);
	}

	//Free all the device memory
	cudaFree(devBuffer);

	printf("Waiting for writing threads to finish...\n");
	//Wait for the pthreads to finish before exiting
	for(int i =0; i < NUM; i++){
		pthread_join(threads[i], NULL);
	}
}

void doJulia(int size){
	//Timers
	struct timeval start, end;
	//Number of iterations per image
	int iterations[] = {20, 25, 30, 35, 40, 50, 100, 200, 400};

	//Array to keep track of pthreads
	pthread_t threads[NUM];

	char out[] = "Julia-0.png";
	int outSize = strlen(out) + 1;

	//Copy size to the constant memory
	HANDLE_ERROR( cudaMemcpyToSymbol(devSize, &size, sizeof(int)) );

	//Allocate the device image buffer
	float *devBuffer;
	HANDLE_ERROR( cudaMalloc( (void **) &devBuffer, size * size * sizeof(float)) );

	//Number of cuda threads to start
	dim3  grid(size, size);

	for(int pos = 0; pos < NUM; pos++){
		printf("Iteration %i\n", pos);

		//6 is the position of the number in the filename
		out[6] = (char) '0' + pos;

		// Create a test image - in this case a Mandelbrot Set fractal
		// The output is a 1D array of floats, length: size * size
		//float *buffer = createMandelbrotImage(size, -0.802, -0.177, 0.011, 100);

		//Start Timer
		gettimeofday(&start, NULL);

		//Copy parameters into device constant memory
		//Need to use a temp variable to hold the parameters (Might be changed to program arguments)
		//Max iteration is in an array
		HANDLE_ERROR( cudaMemcpyToSymbol(devMaxIteration, &(iterations[pos]), sizeof(int)) );

		//Clear device memory
		HANDLE_ERROR( cudaMemset(devBuffer, 0, size * size * sizeof(float)) );


		//Allocate the memory for the image
		float *buffer;
		buffer = (float *) malloc(size * size * sizeof(float));

		//Start the kernel
		j_kernel<<<grid,1>>>(devBuffer);

		//Copy from device
		HANDLE_ERROR( cudaMemcpy( buffer,  devBuffer, size*size*sizeof(float), cudaMemcpyDeviceToHost ) );

		float maxValue = 0;
		for(int i = 0; i < size*size; i++){
			//printf("Buffer value: %f\n", buffer[i]);
			if(buffer[i] > maxValue){
				maxValue = buffer[i];
			}
		}
		//scale between 0-1
		for(int i = 0; i < size*size; i++){
			buffer[i] = buffer[i] / maxValue;
			//Inverse to remove lots of black
			buffer[i] = 1 / buffer[i];
		}

		//End timer
		gettimeofday(&end, NULL);

		printf("Iteration %i time: %lf\n", pos, (end.tv_sec + (end.tv_usec/1000000.0)) - (start.tv_sec + (start.tv_usec/1000000.0)));

		// Save the image to a PNG file
		//Set up the struct
		thread_data_t data;
		data.buffer = buffer;
		//Use the iteration number as TID
		data.tid = pos;
		data.size = size;

		//Have to copy the filename
		data.filename = (char *) malloc( outSize * sizeof(char) );
		strncpy(data.filename, out, outSize);
		//Always make sure to terminate
		data.filename[outSize - 1] = '\0';

		printf("Saving PNG\n\n");
		//Start the pthread
		pthread_create(&(threads[pos]), NULL, writeImage, (void *) &data);

		//pthread_join(threads[pos], NULL);

		//writeImage( (void *) &data);
	}

	//Free all the device memory
	cudaFree(devBuffer);

	printf("Waiting for writing threads to finish...\n");
	//Wait for the pthreads to finish before exiting
	for(int i =0; i < NUM; i++){
		pthread_join(threads[i], NULL);
	}
}


// Creates a test image for saving. Creates a Mandelbrot Set fractal of size size x size
/**
 * CUDA Strategy: The original parameters are never actually changed, so why not move them into the constant memory?
 *
 */
__device__ void createMandelbrotImage(float *devBuffer)
{
   int X = blockIdx.x;
   int Y = blockIdx.y;
   int offset = X+ Y *gridDim.x;

   // Create Mandelbrot set image
   float yP = (yS-rad) + (2.0f*rad/devSize)*Y;

   float xP = (xS-rad) + (2.0f*rad/devSize)*X;

   int iteration = 0;
   float x = 0.0f;
   float y = 0.0f;

   while (x*x + y+y <= 4 && iteration < devMaxIteration)
   {
	   float tmp = x*x - y*y + xP;
	   y = 2.0f*x*y + yP;
	   x = tmp;
	   iteration++;
   }


   if (iteration < devMaxIteration) {
	   float modZ = sqrt(x*x + y*y);
	   float mu = iteration - (log(log(modZ))) / log(2.0f);

	   devBuffer[offset] = mu;
   }
   else {
	   devBuffer[offset] = 0;
   }

   return;
}

/**
 * Julia kernel code
 */
__global__ void j_kernel(float *ptr ){

   int x = blockIdx.x;
   int y = blockIdx.y;

   int index = x + y *gridDim.x ;

   // This always gets a number below 1000, this constantly changes
   //int x = index % size;
   // Int division so this will only change every 1000
   //int y = index / size;

   //printf("Kernel return: %f\n", julia(size, x,y));
   ptr[index] = julia(x,y );
}

__device__ float julia(int x, int y ) {
   //const float scale = 1.5; Switched to define for more speed
   float jx = J_SCALE * (float)(devSize/2 - x)/(devSize/2);
   float jy = J_SCALE * (float)(devSize/2 - y)/(devSize/2);

   cuComplex c(-0.8, 0.156);
   cuComplex a(jx, jy);

   int i = 0;
   for (i=0; i< devMaxIteration; i++) {
	   a = a * a + c;
	   if (a.magnitude2() > 50000)
		   return 0;
	   //return a.magnitude2();
   }

   return a.magnitude2();
}

/*****************************
 * Image Writing Code Below  *
 *****************************
 */

// This function actually writes out the PNG image file
void *writeImage(void *input)
{
	thread_data_t *data = (thread_data_t *) input;
	int tid = data->tid;

	if(WRITE_V){
		printf("Writer thread %i starting\n", tid);
	}
	char* filename = data->filename;
	int size = data->size;
	float *buffer = data->buffer;
	//Title that is written into the file
	char title[] = "Fractal";

	//Timers
	struct timeval start, end;

	gettimeofday(&start, NULL);

	int code = 0;
	FILE *fp;
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep row;

	// Open file for writing (binary mode)
	fp = fopen(filename, "wb");
	if (fp == NULL) {
		fprintf(stderr, "Could not open file %s for writing\n", filename);
		code = 1;
		goto finalise;
	}

	// Initialize write structure
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fprintf(stderr, "Could not allocate write struct\n");
		code = 1;
		goto finalise;
	}

	// Initialize info structure
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fprintf(stderr, "Could not allocate info struct\n");
		code = 1;
		goto finalise;
	}

	// Setup Exception handling
	if (setjmp(png_jmpbuf(png_ptr))) {
		fprintf(stderr, "Error during png creation\n");
		code = 1;
		goto finalise;
	}

	png_init_io(png_ptr, fp);

	// Write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, size, size,
			8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	// Set title
	if (title != NULL) {
		png_text title_text;
		title_text.compression = PNG_TEXT_COMPRESSION_NONE;
		title_text.key = title;//"Title";
		title_text.text = title;
		png_set_text(png_ptr, info_ptr, &title_text, 1);
	}

	png_write_info(png_ptr, info_ptr);

	// Allocate memory for one row (3 bytes per pixel - RGB)
	row = (png_bytep) malloc(3 * size * sizeof(png_byte));

	// Write image data
	int x, y;
	for (y=0 ; y<size ; y++) {
		for (x=0 ; x<size ; x++) {
			setRGB(&(row[x*3]), buffer[y*size + x]);
		}
		png_write_row(png_ptr, row);
	}

	// End write
	png_write_end(png_ptr, NULL);

	finalise:
	if (fp != NULL) fclose(fp);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	if (row != NULL) free(row);

	if(code){
		fprintf(stderr, "Error writing image %i\n", tid);
	}


	if(WRITE_V){
		printf("Writer thread %i ending\n", tid);
	}


	//End timer
	gettimeofday(&end, NULL);

	if(WRITE_V){
		printf("Writing %i time: %lf\n", tid, (end.tv_sec + (end.tv_usec/1000000.0)) - (start.tv_sec + (start.tv_usec/1000000.0)));
	}
	// Free up the memory used to store the image
	//free(buffer);
	//free(filename);
	//free(data);

	//Clean thread exit
	pthread_exit(NULL);
}

// This takes the float value 'val', converts it to red, green & blue values, then
// sets those values into the image memory buffer location pointed to by 'ptr'
inline void setRGB(png_byte *ptr, float val)
{
   int v = (int)(val * 768);
   if (v < 0) v = 0;
   if (v > 768) v = 768;
   int offset = v % 256;

   if (v<256) {
      ptr[0] = 0; ptr[1] = 0; ptr[2] = offset;
   }
   else if (v<512) {
      ptr[0] = 0; ptr[1] = offset; ptr[2] = 255-offset;
   }
   else {
      ptr[0] = offset; ptr[1] = 255-offset; ptr[2] = 0;
   }
}
