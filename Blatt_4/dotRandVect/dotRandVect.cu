#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

float randomNumber(int max)
    {
    return (rand() % (max + 1 ));
    }

struct vect
    {
    float x;
    float y;
    float z;
    };
struct vectProd
    {
    vect v1;
    vect v2;
    float dot;
    };

__global__ void dot(vectProd* pointer)
    {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    pointer[threadId].dot =
        pointer[threadId].v1.x*pointer[threadId].v2.x
        +pointer[threadId].v1.y*pointer[threadId].v2.y
        +pointer[threadId].v1.z*pointer[threadId].v2.z;
    }

void openDotKernelAndTime(int blocksize, int gridsize, vectProd* devicePointer, std::ofstream &f )
    {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 dimBlock1(blocksize);
    dim3 dimGrid1(gridsize);
    cudaEventRecord(start, 0);
    dot<<<dimGrid1, dimBlock1>>>(devicePointer);
    f << "ErrorCode " <<cudaThreadSynchronize()<<std::endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    f << "Blöcke: "<< gridsize << ", Threads: "<< blocksize << std::endl;
    f << ", Timing:" << fixed<< elapsedTime <<std::endl;
    }

int main(int argc, char** argv)
    {
    srand((unsigned) time(NULL));
    vectProd* hostPointer;
    std::ofstream f("timing.txt");
    f << "Timing for different Grid / Block size:" << std::endl << std::endl;
    size_t Elements = 1024*1024;
    size_t sizeInBytes = Elements * sizeof(*hostPointer);
    hostPointer = (vectProd*) malloc(sizeInBytes);
    cudaHostAlloc(&hostPointer, sizeInBytes, cudaHostAllocDefault);
    memset(hostPointer, 0, Elements);

    for ( int l=0; l<Elements; l++ )
        {
        hostPointer[l].v1.x=randomNumber(10);
        hostPointer[l].v1.y=randomNumber(10);
        hostPointer[l].v1.z=randomNumber(10);

        hostPointer[l].v2.x=randomNumber(10);
        hostPointer[l].v2.y=randomNumber(10);
        hostPointer[l].v2.z=randomNumber(10);
        }

    vectProd* devicePointer;
    f << "ErrorCode " <<cudaMalloc(&devicePointer, sizeInBytes)<< std::endl;
    f << "ErrorCode " <<cudaMemcpy(devicePointer, hostPointer, sizeInBytes, cudaMemcpyHostToDevice) <<std::endl;
    f << "ErrorCode "<< cudaThreadSynchronize() << std::endl;

    int blocksize;
    int gridsize;
    //Maximum sizes of each dimension of a block:    1024 x 1024 x 64
    //Maximum sizes of each dimension of a grid:     65535 x 65535 x 1

    blocksize=32;
    gridsize=32768;
    openDotKernelAndTime(blocksize,gridsize, devicePointer, f);

    blocksize=64;
    gridsize=16384;
    openDotKernelAndTime(blocksize,gridsize, devicePointer, f);

    blocksize=128;
    gridsize=8192;
    openDotKernelAndTime(blocksize,gridsize, devicePointer, f);

    blocksize=256;
    gridsize=4096;
    openDotKernelAndTime(blocksize,gridsize, devicePointer, f);

    blocksize=512;
    gridsize=2048;
    openDotKernelAndTime(blocksize,gridsize, devicePointer, f);

    blocksize=1024;
    gridsize=1024;
    openDotKernelAndTime(blocksize,gridsize, devicePointer, f);

    f << "ErrorCode " <<cudaMemcpy(hostPointer, devicePointer, sizeInBytes, cudaMemcpyDeviceToHost)<<std::endl;
    f << "ErrorCode " <<cudaFree(devicePointer)<<std::endl;
    f << "ErrorCode " <<cudaFreeHost(hostPointer)<<std::endl;
    return 0;
    }
