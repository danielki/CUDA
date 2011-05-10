#include <stdio.h>

__global__ void emptyKernel()
    {
    // leerer Kernel
    }

int main(int argc, char ** argv)
    {
    dim3 dimGrid(1);
    dim3 dimBlock(1);
    emptyKernel<<<dimGrid, dimBlock>>>();
    return 0;
    }
