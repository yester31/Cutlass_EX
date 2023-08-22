#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

void check_cuda_version(bool &notSupported)
{
    // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
    // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
    if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0)))
    {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
        notSupported = true;
    }
}

void check_compute_capability(bool &notSupported)
{
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    if (!(props.major >= 8))
    {
        std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80."
                  << std::endl;
        notSupported = true;
    }
}

// Computes the output tensor size (NKPQ)
cutlass::Tensor4DCoord calc_output_size(cutlass::Tensor4DCoord &input_size, cutlass::Tensor4DCoord &padding, cutlass::Tensor4DCoord &filter_size, cutlass::MatrixCoord &conv_stride)
{
    return cutlass::Tensor4DCoord(
        input_size.n(),
        (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
        (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
        filter_size.n());
}

template <
    typename Element_, // Data type of element stored within tensor (concept: NumericType)
    typename Layout_   // Defines a mapping from logical coordinate to linear memory (concept: Layout)
    >
void show_tensor_view(cutlass::HostTensor<Element_, Layout_> &tensor)
{
    std::cout << "tensor size : " << tensor.size() << std::endl;
    std::cout << "tensor shape : " << tensor.extent() << std::endl;
    std::cout << "tensor data type : " << typeid(*tensor.host_data()).name() << std::endl;
    std::cout << "tensor view : " << std::endl;
    std::cout << tensor.host_view() << std::endl;
}

int main(int argc, char const **args)
{
    // 0. cuda & device arch version check
    bool notSupported = false;
    check_cuda_version(notSupported);
    check_compute_capability(notSupported);
    if (notSupported)
        return 0;

    // 1. Allocate host-device tensors using the CUTLASS Utilities.
    cutlass::Tensor4DCoord input_size{1, 4, 4, 3}; // N, H, W, C
    using ElementInputA = cutlass::uint4b_t;       // Data type of elements in input tensor
    using LayoutInputA = cutlass::layout::TensorNHWC;
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(input_size);

    cutlass::Tensor4DCoord filter_size{2, 3, 3, 3}; // K, KH, KW, C
    using ElementInputB = cutlass::half_t;          // Data type of elements in input tensor
    using LayoutInputB = cutlass::layout::TensorNHWC;
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(filter_size);

    cutlass::Tensor4DCoord padding{0, 0, 0, 0}; // T, B, L, R
    cutlass::MatrixCoord conv_stride{1, 1};
    cutlass::Tensor4DCoord output_size = calc_output_size(input_size, padding, filter_size, conv_stride);
    using ElementOutput = float; // Data type of elements in output tensor
    using LayoutOutput = cutlass::layout::TensorNHWC;
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(output_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(output_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(output_size);

    // Initialize tensors

    // Fill tensor A on host with Sequential data
    cutlass::reference::host::BlockFillSequential(tensor_a.host_data(), tensor_a.capacity());
    show_tensor_view(tensor_a);

    // Fill tensor B on host with uniformly distributed random data
    cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1, ElementInputB(7), ElementInputB(-8), 0);
    show_tensor_view(tensor_b);

    // Fill tensor C on host with Sequential data
    cutlass::reference::host::BlockFillSequential(tensor_c.host_data(), tensor_c.capacity());
    show_tensor_view(tensor_c);

    // Fill tensor D on host with zeros
    cutlass::reference::host::TensorFill(tensor_d.host_view());
    show_tensor_view(tensor_d);

    // Fill tensor D for reference on host with zeros
    cutlass::reference::host::TensorFill(tensor_ref_d.host_view());
    show_tensor_view(tensor_ref_d);

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    return 0;
}