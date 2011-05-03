#include "device.h"
#include <fstream>
#include <time.h>
#include <stdio.h>

void   copyToDevice(const primitives &p, float &elapsedTime)
       {
       cudaEvent_t start, stop;
	   cudaEventCreate(&start);
	   cudaEventCreate(&stop);
	   size_t sizeInBytes = sizeof(p);
	   primitives* devicePointer;
	   cudaMalloc(&devicePointer, sizeInBytes);
	   cudaEventRecord(start, 0);
	   cudaMemcpy(devicePointer, &p, sizeInBytes, cudaMemcpyHostToDevice);
	   cudaEventRecord(stop, 0);
	   cudaEventElapsedTime(&elapsedTime, start, stop);
	   cudaEventDestroy(start);
	   cudaEventDestroy(stop);
	   cudaFree(devicePointer);
	   }
	
