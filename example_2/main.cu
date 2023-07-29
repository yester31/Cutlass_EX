#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

int main()
{
    // 4bit data type
    cutlass::uint4b_t i4_a{3};
    cutlass::uint4b_t i4_b{2};

    cutlass::sizeof_bits<cutlass::uint4b_t> sb4;
    std::cout << "sb4.value : " << sb4.value << std::endl;

    cutlass::platform::numeric_limits<cutlass::uint4b_t> nlb4;
    std::cout << "nlb4.max() : " << nlb4.max() << std::endl;

    std::cout << "i4_a + i4_b : " << i4_a + i4_b << std::endl;

    return 0;
}