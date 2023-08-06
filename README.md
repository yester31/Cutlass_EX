# Cutlass_EX

## 0. Introduction
- Goal : Study of Cutlass

## 1. Example List

### example_1) custom code with CUTLASS

### example_2) cutlass::uint4b_t 

### example_3) single-precision gemm template
- [00_basic_gemm](https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu)
- This is kernel computes the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes all matrices have column-major layout.

### example_4) mixed-precision gemm template with cutlass utilities
- [01_cutlass_utilities](https://github.com/NVIDIA/cutlass/blob/main/examples/01_cutlass_utilities/cutlass_utilities.cu)
- These utilities are intended to be useful supporting components for managing tensor and matrix memory allocations, initializing and comparing results, and computing reference output.

### example_5) CUTLASS debugging tool
- [02_dump_reg_shmem](https://github.com/NVIDIA/cutlass/blob/main/examples/02_dump_reg_shmem/dump_reg_shmem.cu)
- Demonstrate CUTLASS debugging tool for dumping fragments and shared memory
- dumping : Record the state of memory at a specific point in time

### example_6) CUTLASS layout visualization example
- [03_visualize_layout](https://github.com/NVIDIA/cutlass/blob/main/examples/03_visualize_layout/visualize_layout.cpp)



## 2. Guide
```
    cd example_{number}
    mkdir build
    cd build
    cmake ..
    make
    ./main
```



## 3 Reference 
* Cutlass : <https://github.com/NVIDIA/cutlass>