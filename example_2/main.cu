#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <string>

int main()
{
    // 4bit data type
    using ui4bit = cutlass::uint4b_t;

    ui4bit ubit4_a{4};
    ui4bit ubit4_b{2};

    // adress
    std::cout << "adress of ubit4_a : " << &ubit4_a << std::endl;
    std::cout << "adress of ubit4_b : " << &ubit4_b << std::endl;

    // array
    int arr0[]{1, 2, 3, 4, 5};
    std::cout << "adress of int arr0[0] : " << &arr0[0] << std::endl;
    std::cout << "adress of int arr0[1] : " << &arr0[1] << std::endl;
    std::cout << "adress of int arr0[2] : " << &arr0[2] << std::endl;
    std::cout << "adress of int arr0[3] : " << &arr0[3] << std::endl;
    std::cout << "adress of int arr0[4] : " << &arr0[4] << std::endl;

    cutlass::half_t arr1[]{1_hf, 2_hf, 3_hf, 4_hf, 5_hf};
    std::cout << "adress of half_t arr1[0] : " << &arr1[0] << std::endl;
    std::cout << "adress of half_t arr1[1] : " << &arr1[1] << std::endl;
    std::cout << "adress of half_t arr1[2] : " << &arr1[2] << std::endl;
    std::cout << "adress of half_t arr1[3] : " << &arr1[3] << std::endl;
    std::cout << "adress of half_t arr1[4] : " << &arr1[4] << std::endl;

    // std::uint8_t arr2[]{1, 2, 3, 4, 5};
    // std::cout << "adress of uint8_t arr2[0] : " << &arr2[0] << std::endl;
    // std::cout << "adress of uint8_t arr2[1] : " << &arr2[1] << std::endl;
    // std::cout << "adress of uint8_t arr2[2] : " << &arr2[2] << std::endl;
    // std::cout << "adress of uint8_t arr2[3] : " << &arr2[3] << std::endl;
    // std::cout << "adress of uint8_t arr2[4] : " << &arr2[4] << std::endl;

    ui4bit arr3[]{1, 2, 3, 4, 5};
    std::cout << "adress of uint4b_t arr3[0] : " << &arr3[0] << std::endl;
    std::cout << "adress of uint4b_t arr3[1] : " << &arr3[1] << std::endl;
    std::cout << "adress of uint4b_t arr3[2] : " << &arr3[2] << std::endl;
    std::cout << "adress of uint4b_t arr3[3] : " << &arr3[3] << std::endl;
    std::cout << "adress of uint4b_t arr3[4] : " << &arr3[4] << std::endl;

    // data size check
    cutlass::sizeof_bits<ui4bit> sbit4;
    std::cout << "uint4b_t bit size : " << sbit4.value << std::endl;
    std::cout << "real storage size : " << sizeof(sbit4) << std::endl;

    // numeric range
    cutlass::platform::numeric_limits<ui4bit> nlb4;
    std::cout << "uint4b_t min : " << nlb4.lowest() << std::endl;
    std::cout << "uint4b_t max : " << nlb4.max() << std::endl;
    std::cout << "uint4b_t is_integer : " << nlb4.is_integer << std::endl;

    // simple numeric calculate
    std::cout << "ubit4_a(" + std::to_string(ubit4_a) + ") + ubit4_b(" + std::to_string(ubit4_b) + ") : " << ubit4_a + ubit4_b << std::endl;
    std::cout << "ubit4_a(" + std::to_string(ubit4_a) + ") - ubit4_b(" + std::to_string(ubit4_b) + ") : " << ubit4_a - ubit4_b << std::endl;
    std::cout << "ubit4_a(" + std::to_string(ubit4_a) + ") * ubit4_b(" + std::to_string(ubit4_b) + ") : " << ubit4_a * ubit4_b << std::endl;
    std::cout << "ubit4_a(" + std::to_string(ubit4_a) + ") / ubit4_b(" + std::to_string(ubit4_b) + ") : " << ubit4_a / ubit4_b << std::endl;

    return 0;
}