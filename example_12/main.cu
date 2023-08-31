#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
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

// Compute performance in Gflop/s
// Gflop/s stands for billions (10^9) of
// floating-point operations per second (Gflop/s).
double gflops(double runtime_s, cutlass::Tensor4DCoord &output_size, cutlass::Tensor4DCoord &filter_size)
{

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size.product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
}

int main(int argc, char const **args)
{
    //
    // 0. cuda & device arch version check
    //
    bool notSupported = false;
    check_cuda_version(notSupported);
    check_compute_capability(notSupported);
    if (notSupported)
        return 0;

    //
    // 1. Allocate host-device tensors using the CUTLASS Utilities.
    //
    using DataType = cutlass::half_t; // Data type of elements in output tensor
    using TensorLayout = cutlass::layout::TensorNHWC;
    using ElementAccumulator = float;     // Data type of accumulator
    using ElementComputeEpilogue = float; // Data type of epilogue computation (alpha, beta)

    cutlass::Tensor4DCoord input_size{1, 64, 64, 32}; // N, H, W, C
    cutlass::HostTensor<DataType, TensorLayout> input_tensor(input_size);

    cutlass::Tensor4DCoord filter_size{64, 3, 3, 32}; // K, KH, KW, C
    cutlass::HostTensor<DataType, TensorLayout> filter_tensor(filter_size);

    cutlass::Tensor4DCoord padding{0, 0, 0, 0}; // T, B, L, R
    cutlass::MatrixCoord conv_stride{1, 1};
    cutlass::MatrixCoord dilation{1, 1};
    cutlass::Tensor4DCoord output_size = calc_output_size(input_size, padding, filter_size, conv_stride);
    std::cout << "output_size : " << output_size << std::endl;
    cutlass::HostTensor<DataType, TensorLayout> output_tensor(output_size);
    cutlass::HostTensor<DataType, TensorLayout> tensor_d(output_size);
    cutlass::HostTensor<DataType, TensorLayout> tensor_ref_d(output_size);

    //
    // 2. Initialize tensors
    //
    // Fill tensor input_tensor with Sequential numbers
    // cutlass::reference::host::BlockFillSequential(input_tensor.host_data(), input_tensor.capacity());
    cutlass::reference::host::TensorFill(input_tensor.host_view(), cutlass::half_t(2));
    // show_tensor_view(input_tensor);

    // Fill tensor output_tensor with ones
    cutlass::reference::host::TensorFill(filter_tensor.host_view(), cutlass::half_t(1));
    // show_tensor_view(filter_tensor);

    // Fill tensor output_tensor with zeros
    cutlass::reference::host::TensorFill(output_tensor.host_view());
    // show_tensor_view(output_tensor);
    cutlass::reference::host::TensorFill(tensor_d.host_view());
    // show_tensor_view(tensor_d);
    cutlass::reference::host::TensorFill(tensor_ref_d.host_view());
    // show_tensor_view(tensor_ref_d);

    //
    // 3. Compute reference implementation on host side
    //
    // Construct Conv2dProblemSize with user defined output size
    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        cutlass::conv::Mode::kCrossCorrelation,
        1 // Split K dimension into 1 partitions
    );

    ElementComputeEpilogue alpha{1};
    ElementComputeEpilogue beta{0};

    std::cout << "Conv2d on host...\n";
    cutlass::reference::host::Conv2dFprop<
        DataType, TensorLayout,
        DataType, TensorLayout,
        DataType, TensorLayout,
        ElementAccumulator,
        ElementComputeEpilogue>(
        problem_size,
        input_tensor.host_ref(),
        filter_tensor.host_ref(),
        output_tensor.host_ref(),
        tensor_ref_d.host_ref(),
        alpha,
        beta);

    // show_tensor_view(tensor_ref_d);

    //
    // 4. Copy data from host to GPU
    //
    input_tensor.sync_device();
    filter_tensor.sync_device();
    output_tensor.sync_device();
    tensor_d.sync_device();

    //
    // 5. define kernel properties type
    //
    // Whether to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;
    // SM architecture number
    using SmArch = cutlass::arch::Sm86;
    // Threadblock tile shape
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    // Warp tile shape
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    // MMA (Tensor Core instruction, in this case) tile shape
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    // How the kernel schedules threadblocks
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    // Number of pipeline stages to use
    constexpr int NumStages = 3;
    // Which iterator algorithm to use: Analytic or Optimized
    static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;
    // The epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        DataType,                                    // Data type of output matrix.
        128 / cutlass::sizeof_bits<DataType>::value, // The number of elements per vectorized
                                                     // memory access. This becomes the vector width of
                                                     // math instructions in the epilogue too.
        ElementAccumulator,                          // Data type of accumulator
        ElementComputeEpilogue>;                     // Data type for alpha/beta in linear combination
    // Kernel properties type
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        DataType, TensorLayout,
        DataType, TensorLayout,
        DataType, TensorLayout,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        cutlass::arch::OpMultiplyAdd,
        IteratorAlgorithm>::Kernel;

    // Type of the actual kernel
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    // define arguments for CUTLASS Convolution
    // Construct ImplicitGemm::Argument structure with conv2d problem size, data pointers, and epilogue values
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        input_tensor.device_ref(),
        filter_tensor.device_ref(),
        output_tensor.device_ref(),
        tensor_d.device_ref(),
        {alpha, beta},
    };

    //
    // 6. Initialize CUTLASS Convolution
    //
    cutlass::Status status;
    ImplicitGemm implicit_gemm_op;
    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    //
    // 7. Launch CUTLASS kernel
    //
    std::cout << "Conv2d on device using cutlass kernel...\n";
    status = implicit_gemm_op();

    CUTLASS_CHECK(status);

    //
    // 8. Check if CUTLASS kernel and reference kernel produced the same output
    //
    tensor_d.sync_host(); // cutlass output

    bool passed = cutlass::reference::host::TensorEquals(tensor_d.host_view(), tensor_ref_d.host_view());

    if (!passed)
    {
        status = cutlass::Status::kErrorInternal;
        std::cout << "ERROR - results miscompared.\n";
    }
    else
    {
        status = cutlass::Status::kSuccess;
        std::cout << "Passed.\n";
    }
    CUTLASS_CHECK(status);
    // show_tensor_view(tensor_d);

    //
    // 9. Performance measurement
    //
    cudaEvent_t events[2];
    cudaError_t error;

    for (auto &event : events)
    {
        error = cudaEventCreate(&event);
        if (error != cudaSuccess)
        {
            std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(error) << std::endl;
        }
    }

    // Record an event at the start of a series of convolution operations.
    error = cudaEventRecord(events[0]);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
    }

    // Launch a sequence of implicit GEMM operations on the device.
    int iterations = 20;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        status = implicit_gemm_op();
        CUTLASS_CHECK(status);
    }

    // Record an event when the convolutions have been launched.
    error = cudaEventRecord(events[1]);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
    }

    // Wait for work on the device to complete.
    error = cudaEventSynchronize(events[1]);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(error) << std::endl;
    }

    // Measure elapsed runtime.
    float runtime_ms = 0;
    error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(error) << std::endl;
    }

    // Print average run time and floating-point throughput (Gflop/s).
    runtime_ms = double(runtime_ms) / double(iterations);
    double gflops_v = gflops(runtime_ms / 1000.0, output_size, filter_size);
    std::cout << "runtime : " << runtime_ms << "[ms], gflops : " << gflops_v << std::endl;

    // Cleanup
    for (auto event : events)
    {
        (void)cudaEventDestroy(event);
    }

    return 0;
}