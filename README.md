# Cutlass_EX

## 0. Introduction
- Goal : Study of Cutlass

## 1. Example List

### 01) custom code with CUTLASS

### 02) cutlass::uint4b_t 

### 03) single-precision gemm template
- [00_basic_gemm](https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu)
- This is kernel computes the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes all matrices have column-major layout.

### 04) mixed-precision gemm template with cutlass utilities
- [01_cutlass_utilities](https://github.com/NVIDIA/cutlass/blob/main/examples/01_cutlass_utilities/cutlass_utilities.cu)
- These utilities are intended to be useful supporting components for managing tensor and matrix memory allocations, initializing and comparing results, and computing reference output.


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