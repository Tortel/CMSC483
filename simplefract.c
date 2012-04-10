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




// Creates a test image for saving. Creates a Mandelbrot Set fractal of size size x size
float *createMandelbrotImage(int size, float xS, float yS, float rad, int maxIteration);

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
   for(int pos = 0; pos < NUM; pos++){
      printf("Iteration %i\n", pos);

      //11 is the position of the number in the filename
      out[11] = (char) '0' + pos;

      // Create a test image - in this case a Mandelbrot Set fractal
      // The output is a 1D array of floats, length: size * size
      //float *buffer = createMandelbrotImage(size, -0.802, -0.177, 0.011, 100);

      //Start Timer
      gettimeofday(&start, NULL);

      float *buffer = createMandelbrotImage(size, -0.802, -0.177, 0.011, iterations[pos]);
      if (buffer == NULL) {
         return 1;
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
      pthread_create(&threads[pos], NULL, writeImage, (void *) &data);

      //writeImage( (void *) &data);
   }


   printf("Waiting for writing threads to finish...\n");
   //Wait for the pthreads to finish before exiting
   for(int i =0; i < NUM; i++){
	   pthread_join(threads[i], NULL);
   }

   printf("Done!\n");
   return 0;
}

// Creates a test image for saving. Creates a Mandelbrot Set fractal of size size x size
float *createMandelbrotImage(int size, float xS, float yS, float rad, int maxIteration)
{
   float *buffer = (float *) malloc(size * size * sizeof(float));
   if (buffer == NULL) {
      fprintf(stderr, "Could not create image buffer\n");
      return NULL;
   }

   // Create Mandelbrot set image

   int xPos, yPos;
   float minMu = maxIteration;
   float maxMu = 0;

   for (yPos=0 ; yPos<size ; yPos++)
   {
      float yP = (yS-rad) + (2.0f*rad/size)*yPos;

      for (xPos=0 ; xPos<size ; xPos++)
      {
         float xP = (xS-rad) + (2.0f*rad/size)*xPos;

         int iteration = 0;
         float x = 0;
         float y = 0;

         while (x*x + y+y <= 4 && iteration < maxIteration)
         {
            float tmp = x*x - y*y + xP;
            y = 2*x*y + yP;
            x = tmp;
            iteration++;
         }

         if (iteration < maxIteration) {
            float modZ = sqrt(x*x + y*y);
            float mu = iteration - (log(log(modZ))) / log(2);
            if (mu > maxMu) maxMu = mu;
            if (mu < minMu) minMu = mu;
            buffer[yPos * size + xPos] = mu;
         }
         else {
            buffer[yPos * size + xPos] = 0;
         }
      }
   }

   // Scale buffer values between 0 and 1
   int count = size * size;
   while (count) {
      count --;
      buffer[count] = (buffer[count] - minMu) / (maxMu - minMu);
      printf("Buffer[%i]: %f \n", count, buffer[count]);
   }

   printf("minMu: %f\n", minMu);
   printf("maxMu: %f\n", maxMu);

   return buffer;
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

	if(DEBUG){
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
		fprintf(stderr, "Error writing image %i\n", tid);
	}

	if(DEBUG){
		printf("Writer thread %i ending\n", tid);
	}

	//End timer
	gettimeofday(&end, NULL);

	printf("Writing %i time: %lf\n", tid, (end.tv_sec + (end.tv_usec/1000000.0)) - (start.tv_sec + (start.tv_usec/1000000.0)));

	// Free up the memory used to store the image
	free(buffer);
	free(filename);
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
