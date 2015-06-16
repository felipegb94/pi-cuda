/* Pi - CUDA version 1 - uses integers for CUDA kernels
 * Author: Aaron Weeden, Shodor, May 2015
 */
#include <stdio.h> /* fprintf() */
#include <float.h> /* DBL_EPSILON() */
#include <math.h> /* sqrt() */

__global__ void calculateAreas(int offset, const int numRects, const double width,
    double *dev_areas) {
  const int threadId = threadIdx.x + offset;
  const double x = (threadId * width);
  const double heightSq = (1.0 - (x * x));
  const double height =
    /* Prevent nan value for sqrt() */
    (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));

  if (threadId < numRects) {
    dev_areas[threadId] = (width * height);
  }
}

void calculateArea(const int numRects, double *area) {
  double *areas = (double*)malloc(numRects * sizeof(double));
  double *dev_areas;
  int i = 0;
  cudaError_t err;

  if (areas == NULL) {
    fprintf(stderr, "malloc failed!\n");
  }

  err = cudaMalloc((void**)&dev_areas, (numRects * sizeof(double)));

  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
  }
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device); 
  printf("Max number of threads per block = %i \n", prop.maxThreadsPerBlock);
  if(numRects > prop.maxThreadsPerBlock){
  	int numPasses = ceil((double) (numRects/prop.maxThreadsPerBlock));
	for(int i = 0; i < numPasses; i++){
		int offset = (prop.maxThreadsPerBlock * i);
		calculateAreas<<<1, numRects>>>(offset, numRects, (1.0 / numRects), dev_areas);
	}
  }
  else{
        calculateAreas<<<1, numRects>>>(0, numRects, (1.0 / numRects), dev_areas);
  }

  err = cudaMemcpy(areas, dev_areas, (numRects * sizeof(double)),
    cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
  }

  (*area) = 0.0;
  for (i = 0; i < numRects; i++) {
    (*area) += areas[i];
  }

  cudaFree(dev_areas);

  free(areas);
}
