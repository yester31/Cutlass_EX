# Cutlass_EX

## 0. Introduction
- Goal : Development of a 4-bit primitives by using Cutlass

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

### example_7) CUTLASS example to compute a batched strided gemm in two different ways
- [05_batched_gemm](https://github.com/NVIDIA/cutlass/blob/main/examples/05_batched_gemm/batched_gemm.cu)
- strided batched gemm : By specifying pointers to the first matrices of the batch and the stride between the consecutive matrices of the batch.
- array gemm : By copying pointers to all matrices of the batch to the device memory.


### example_8) CUTLASS turing gemm using tensor cores
- [08_turing_tensorop_gemm](https://github.com/NVIDIA/cutlass/blob/main/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu)


### example_9) CUTLASS turing convolution using tensor cores
- [09_turing_tensorop_conv2dfprop](https://github.com/NVIDIA/cutlass/blob/main/examples/09_turing_tensorop_conv2dfprop/turing_tensorop_conv2dfprop.cu)



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