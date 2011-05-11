cmake_minimum_required(VERSION 2.8.0)

# for finding GLog
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIR})

find_package(GLog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

find_package(CUDA REQUIRED)

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(YAMLCPP REQUIRED)
include_directories(${YAMLCPP_INCLUDE_DIR})

include_directories("src")
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS "-arch=sm_20" )
add_executable(rt src/main.cc src/parser.cc src/raytracer.cc src/ppm_writer.cc src/types.cc)
set_target_properties(rt PROPERTIES COMPILE_FLAGS "-D__CPU__")
target_link_libraries(rt ${GLOG_LIBRARY})
target_link_libraries(rt ${GFLAGS_LIBRARY})
target_link_libraries(rt ${YAMLCPP_LIBRARY})

CONFIGURE_FILE(src/raytracer.cc src/raytracer.cu COPYONLY)
cuda_add_executable(rtcuda src/main.cc src/parser.cc src/raytracer.cu src/ppm_writer.cc src/types.cc OPTIONS "-D__CUDA__")
set_target_properties(rtcuda PROPERTIES COMPILE_FLAGS "-D__CUDA__")
target_link_libraries(rtcuda ${GLOG_LIBRARY})
target_link_libraries(rtcuda ${GFLAGS_LIBRARY})
target_link_libraries(rtcuda ${YAMLCPP_LIBRARY})



# common flags
SET(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas -Wno-long-long -Wno-variadic-macros")
SET(CMAKE_CXX_FLAGS_RELEASE "-DNEXPENSIVE -O2 -g0")
SET(CMAKE_CXX_FLAGS_DEBUG "-DWARNINGS -O0 -g3 -fno-inline")

