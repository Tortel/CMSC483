/*
Sample taken from here:
http://www.labbookpages.co.uk/software/imgProc/libPNG.html
*/

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <cuda.h>

//Number of images/threads
#define NUM 9

//Debugging printing
#define DEBUG 1

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

//Device buffer
__device__ float *devBuffer;

//Min/max mu, used for colors (I think?)
__device__ float minMu;
__device__ float maxMu;
//Temp vars, to use with the atomic max/min
__device__ unsigned int tMinMu;
__device__ unsigned int tMaxMu;


// Creates a test image for saving. Creates a Mandelbrot Set fractal of size size x size
__global__ void createMandelbrotImage();


/**
 * Functions to convert a float into an unsinged int
 */
__device__ inline unsigned int FloatFlip(unsigned int f)
{
   unsigned int mask = -int(f >> 31) | 0x80000000;
   return f ^ mask;
}

__device__ inline unsigned int IFloatFlip(unsigned int f)
{
   unsigned int mask = ((f >> 31) - 1) | 0x80000000;
   return f ^ mask;
}


// This takes the float value 'val', converts it to red, green & blue values, then
// sets those values into the image memory buffer location pointed to by 'ptr'
inline void setRGB(png_byte *ptr, float val);

// This function actually writes out the PNG image file
void *writeImage( void *input);


int main(int argc, char *argv[])
{
   //Image size as first parameter
   int size;
   if(argc == 2){
      size = atoi(argv[1]);
   } else{
      size = 1000;
   }
   //Timers
   struct timeval start, end;
   //Number of iterations per image
   int iterations[] = {20, 25, 30, 35, 40, 50, 100, 300, 500, 1000};

   //Array to keep track of pthreads
   pthread_t threads[NUM];

   char out[] = "Mandelbrot-0.png";
   int outSize = strlen(out) + 1;

   //Copy size to the constant memory
   cudaMemcpyToSymbol(devSize, &size, sizeof(int));

   //Allocate the device image buffer
   cudaMalloc( (void **) &devBuffer, size * size * sizeof(float));

   //Number of cuda threads to start
   dim3  grid(size, size);

   //Temp var for parameters
   float *tmp;
   tmp = (float *) malloc(sizeof(float));

   for(int pos = 0; pos < NUM; pos++){
      printf("Iteration %i\n", pos);

      //11 is the position of the number in the filename
      out[11] = (char) '0' + pos;

      // Create a test image - in this case a Mandelbrot Set fractal
      // The output is a 1D array of floats, length: size * size
      //float *buffer = createMandelbrotImage(size, -0.802, -0.177, 0.011, 100);

      //Start Timer
      gettimeofday(&start, NULL);

      //Copy parameters into device constant memory
      //Need to use a temp variable to hold the parameters (Might be changed to program arguments)
      *tmp = -0.802f;
      cudaMemcpyToSymbol(xS, tmp, sizeof(float));
      *tmp = -0.177f;
      cudaMemcpyToSymbol(yS, tmp, sizeof(float));
      *tmp = 0.011f;
      cudaMemcpyToSymbol(rad, tmp, sizeof(float));
      //Max iteration is in an array
      cudaMemcpyToSymbol(devMaxIteration, &(iterations[pos]), sizeof(int));

      //Set up min/max mu
      *tmp = 0;
      cudaMemcpy(tmp, &minMu, sizeof(float), cudaMemcpyHostToDevice);
      *tmp = (float) iterations[pos];
      cudaMemcpy(tmp, &maxMu, sizeof(float), cudaMemcpyHostToDevice);


      //Allocate the memory for the image
      float *buffer;
      buffer = (float *) malloc(size * size * sizeof(float));
      //Note: Dont need to copy to device, just let it be overwritten.

      //Start the kernel
      createMandelbrotImage<<<1,grid>>>();

      //Copy from device
      cudaMemcpy(buffer, devBuffer, size * size * sizeof(float), cudaMemcpyDeviceToHost);

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
      pthread_create(&threads[pos], NULL, writeImage, (void *) &data);

      pthread_join(threads[pos], NULL);

      //writeImage( (void *) &data);
   }

   //Free all the device memory
   cudaFree(devBuffer);

   printf("Waiting for writing threads to finish...\n");
   //Wait for the pthreads to finish before exiting
   for(int i =0; i < NUM; i++){
	   pthread_join(threads[i], NULL);
   }

   printf("Done!\n");
   return 0;
}

// Creates a test image for saving. Creates a Mandelbrot Set fractal of size size x size
/**
 * CUDA Strategy: The original parameters are never actually changed, so why not move them into the constant memory?
 *
 */
__device__ void createMandelbrotImage()
{
   // Create Mandelbrot set image
   float yP = (yS-rad) + (2.0f*rad/devSize)*threadIdx.y;

   float xP = (xS-rad) + (2.0f*rad/devSize)*threadIdx.x;

   int iteration = 0;
   float x = 0;
   float y = 0;

   while (x*x + y+y <= 4 && iteration < devMaxIteration)
   {
	   float tmp = x*x - y*y + xP;
	   y = 2*x*y + yP;
	   x = tmp;
	   iteration++;
   }

   __syncthreads();

   //Single thread does this
   if(threadIdx.x ==1 && threadIdx.y == 1){
      tMaxMu = FloatFlip( 0 );
      tMinMu = FloatFlip(devMaxIteration);
   }

   __syncthreads();

   if (iteration < devMaxIteration) {
	   float modZ = sqrt(x*x + y*y);
	   float mu = iteration - (log2(log2(modZ))) / log2(2.0f);
	   unsigned int tMu = FloatFlip(mu);

	   atomicMax(&tMaxMu, tMu);
	   atomicMin(&tMinMu, tMu);
	   /**
	    * http://forums.nvidia.com/index.php?showtopic=91491
	    */
	   //if(mu > maxMu) atomicExch(&maxMu, mu);//atomicMax( &maxMu, mu); //if (mu > maxMu) maxMu = mu;
	   //if(mu < minMu) atomicExch(&minMu, mu);//atomicMin( &minMu, mu); //if (mu < minMu) minMu = mu;
	   devBuffer[threadIdx.y * devSize + threadIdx.x] = mu;
   }
   else {
	   devBuffer[threadIdx.y * devSize + threadIdx.x] = 0;
   }

   __syncthreads();

   //Single thread
   if(threadIdx.x == 1 && threadIdx.y == 1){
      maxMu = IFloatFlip(tMaxMu);
      minMu = IFloatFlip(tMinMu);
   }

   __syncthreads();
   // Scale buffer values between 0 and 1
   //int count = devSize * devSize;
   //while (count) {
      //count --;
   devBuffer[threadIdx.y * devSize + threadIdx.x] = (devBuffer[threadIdx.y + threadIdx.x] - minMu) / (maxMu - minMu);
   //}

   return;
}

/*****************************
 * Image Writing Code Below  *
 *****************************
 */

// This function actually writes out the PNG image file
void *writeImage(void *input)
{
	thread_data_t *data = (thread_data_t *) input;

	printf("Writer thread %i starting\n", data->tid);
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
		title_text.key = "Title";
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
		fprintf(stderr, "Error writing image %i\n", data->tid);
	}


	printf("Writer thread %i ending\n", data->tid);


	//End timer
	gettimeofday(&end, NULL);

	printf("Writing %i time: %lf\n", data->tid, (end.tv_sec + (end.tv_usec/1000000.0)) - (start.tv_sec + (start.tv_usec/1000000.0)));

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
