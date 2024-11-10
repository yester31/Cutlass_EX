cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(main LANGUAGES CXX CUDA)

#
# CUTLASS 3.x requires C++17
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# cutlass
include_directories(/home/pyh/playground/cutlass/include)

add_executable(main main.cu)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES "86")

message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}") 
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}") 
