#include <glog/logging.h>

#include "types.h"
#include "ppm_writer.h"
#include "raytracer.h"

#if __CUDA__

texture<rgb, 2, cudaReadModeElementType> textureRef;

__global__ void textureAntialiasKernel(rgb* ptr, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		rgb res1 = tex2D(textureRef, 2*x, 2*y);
		rgb res2 = tex2D(textureRef, 2*x+1, 2*y);
		rgb res3 = tex2D(textureRef, 2*x, 2*y+1);
		rgb res4 = tex2D(textureRef, 2*x+1, 2*y+1);
		ptr[x+y*width].x = (res1.x + res2.x + res3.x + res4.x )/ 4;
		ptr[x+y*width].y = (res1.y + res2.y + res3.y + res4.y )/ 4;
		ptr[x+y*width].z = (res1.z + res2.z + res3.z + res4.z )/ 4;
		ptr[x+y*width].w = 0;
	}
}
#endif
void antialiasing(scene s, int imageWidth, int imageHeight, char* filename)
    {
#if __CUDA__
	// image dimensions
	int textureImageWidth = imageWidth*2;
	int textureImageHeight = imageHeight*2;
	rgb *imagePtr = (rgb*)malloc(sizeof(rgb) * textureImageWidth * textureImageHeight);
	CHECK_NOTNULL(imagePtr);
	rgb *anti_imagePtr = (rgb*)malloc(sizeof(rgb) * imageWidth * imageHeight);
	CHECK_NOTNULL(anti_imagePtr);
	
	// scene mit großer Auflösung rendern
	render_image(s, textureImageHeight, textureImageWidth, &imagePtr[0]);

	size_t texturePitch;
	
	rgb* devImagePointer; 
	cudaError_t error = cudaMalloc(&devImagePointer, sizeof(rgb)*imageWidth*imageHeight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(devImagePointer);
    
	rgb* textureImageDevicePtr;
	error = cudaMallocPitch(&textureImageDevicePtr, &texturePitch, 
	(sizeof(rgb) *	textureImageWidth), textureImageHeight);
	CHECK_NOTNULL(textureImageDevicePtr);					
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);


	error = cudaMemcpy2D(textureImageDevicePtr, texturePitch,
		                     &imagePtr[0], sizeof(rgb) * textureImageWidth,
		                     sizeof(rgb) * textureImageWidth, textureImageHeight,
		                     cudaMemcpyHostToDevice);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();


	error = cudaBindTexture2D(NULL, &textureRef, textureImageDevicePtr, &channelDesc,
			                  textureImageWidth, textureImageHeight, texturePitch);
	
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);


	dim3 dimBlock(16, 16);
	dim3 dimGrid((imageWidth + dimBlock.x - 1) / dimBlock.x,
	             (imageHeight + dimBlock.y - 1) / dimBlock.y);

	cudaGetLastError();

	textureAntialiasKernel<<<dimGrid, dimBlock>>>(devImagePointer, imageWidth, imageHeight);

	error = cudaGetLastError();
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	
	error = cudaThreadSynchronize();
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

	error = cudaUnbindTexture(textureRef);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

	error = cudaMemcpy(anti_imagePtr,devImagePointer, sizeof(rgb) * imageWidth*imageHeight,cudaMemcpyDeviceToHost);
	                     
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	char new_filename[30] = "anti_";
    strcat( new_filename, filename );
    write_texture_ppm(&anti_imagePtr[0],imageWidth, imageHeight, new_filename , 255);
    
	error = cudaFree(devImagePointer);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    free(imagePtr);
	free(anti_imagePtr); 

#endif
#if __CPU__
    // mache vorerst nichts
#endif
	}
	
	
