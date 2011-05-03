cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
cuda_add_executable(dotRandVect dotRandVect.cu)
