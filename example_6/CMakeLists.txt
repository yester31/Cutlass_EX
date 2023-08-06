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
include_directories(/home/yhpark/workspace/cutlass/include)
include_directories(/home/yhpark/workspace/cutlass/tools/util/include)
include_directories(/home/yhpark/workspace/cutlass/examples/common)

add_library(plugin SHARED register_layout.cu)
target_link_libraries(plugin cudart)

add_executable(main visualize_layout.cpp)

target_link_libraries(main plugin)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES "86")

message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}") 
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}") 
