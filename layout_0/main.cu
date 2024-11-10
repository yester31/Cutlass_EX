#include <iostream>
#include <cutlass/cutlass.h>
#include <cute/layout.hpp>

using namespace cute;

template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const &layout)
{
    for (int m = 0; m < size<0>(layout); ++m)
    {
        for (int n = 0; n < size<1>(layout); ++n)
        {
            printf("%3d  ", layout(m, n));
        }
        printf("\n");
    }
    printf("======================\n");
};

template <class Shape, class Stride>
void print1D(Layout<Shape, Stride> const &layout)
{
    for (int i = 0; i < size(layout); ++i)
    {
        printf("%3d  ", layout(i));
    }
    printf("\n======================\n");
};

int main()
{
    Layout s8 = make_layout(Int<8>{});
    print1D(s8);
    //  0    1    2    3    4    5    6    7

    Layout d8 = make_layout(8);
    print1D(d8);
    //  0    1    2    3    4    5    6    7

    Layout s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
    print1D(s2xs4);
    //  0    1    2    3    4    5    6    7
    print2D(s2xs4);
    //   0    2    4    6
    //   1    3    5    7
    print_layout(s2xs4);
    // (_2,_4):(_1,_2)
    //       0   1   2   3
    //     +---+---+---+---+
    //  0  | 0 | 2 | 4 | 6 |
    //     +---+---+---+---+
    //  1  | 1 | 3 | 5 | 7 |
    //     +---+---+---+---+

    Layout s2xd4 = make_layout(make_shape(Int<2>{}, 4));
    std::cout << "-> s2xd4" << std::endl;
    print_layout(s2xd4);
    //     (_2,4):(_1,_2)
    //       0   1   2   3
    //     +---+---+---+---+
    //  0  | 0 | 2 | 4 | 6 |
    //     +---+---+---+---+
    //  1  | 1 | 3 | 5 | 7 |
    //     +---+---+---+---+

    Layout s2xd4_col = make_layout(make_shape(Int<2>{}, 4), LayoutLeft{});
    std::cout << "-> s2xd4_col" << std::endl;
    print_layout(s2xd4_col);
    // (_2,4):(_1,_2)
    //       0   1   2   3
    //     +---+---+---+---+
    //  0  | 0 | 2 | 4 | 6 |
    //     +---+---+---+---+
    //  1  | 1 | 3 | 5 | 7 |
    //     +---+---+---+---+

    Layout s2xd4_row = make_layout(make_shape(Int<2>{}, 4), LayoutRight{});
    std::cout << "-> s2xd4_row" << std::endl;
    print_layout(s2xd4_row);
    // (_2,4):(4,_1)
    //       0   1   2   3
    //     +---+---+---+---+
    //  0  | 0 | 1 | 2 | 3 |
    //     +---+---+---+---+
    //  1  | 4 | 5 | 6 | 7 |
    //     +---+---+---+---+

    Layout s2xd4_a = make_layout(make_shape(Int<2>{}, 4), make_stride(Int<12>{}, Int<1>{}));
    std::cout << "-> s2xd4_a" << std::endl;
    print1D(s2xd4_a);
    //   0   12    1   13    2   14    3   15
    print_layout(s2xd4_a);
    // (_2,4):(_12,_1)
    //        0    1    2    3
    //     +----+----+----+----+
    //  0  |  0 |  1 |  2 |  3 |
    //     +----+----+----+----+
    //  1  | 12 | 13 | 14 | 15 |
    //     +----+----+----+----+

    Layout s2xh4 = make_layout(make_shape(2, make_shape(2, 2)), make_stride(4, make_stride(2, 1)));
    std::cout << "-> s2xh4" << std::endl;
    print1D(s2xh4);
    //   0    4    2    6    1    5    3    7
    print_layout(s2xh4);
    // (2,(2,2)):(4,(2,1))
    //       0   1   2   3
    //     +---+---+---+---+
    //  0  | 0 | 2 | 1 | 3 |
    //     +---+---+---+---+
    //  1  | 4 | 6 | 5 | 7 |
    //     +---+---+---+---+

    Layout s2xh4_col = make_layout(shape(s2xh4), LayoutLeft{});
    std::cout << "-> s2xh4_col" << std::endl;
    print1D(s2xh4_col);
    //   0    1    2    3    4    5    6    7
    print_layout(s2xh4_col);

    Layout tt = make_layout(make_shape(4, 2), make_stride(2, 3));
    print1D(tt);
    //   0    2    4    6    3    5    7    9
    print2D(tt);
    //   0    3
    //   2    5
    //   4    7
    //   6    9
    print_layout(tt);
    // (4,2):(2,3)
    //        0    1
    //     +----+----+
    //  0  |  0 |  3 |
    //     +----+----+
    //  1  |  2 |  5 |
    //     +----+----+
    //  2  |  4 |  7 |
    //     +----+----+
    //  3  |  6 |  9 |
    //     +----+----+

    Layout tt2 = make_layout(make_shape(4, make_shape(2, 2)), make_stride(2, make_stride(1, 8)));
    print1D(tt2);
    //  0    2    4    6    1    3    5    7    8   10   12   14    9   11   13   15
    print2D(tt2);
    //   0    1    8    9
    //   2    3   10   11
    //   4    5   12   13
    //   6    7   14   15
    print_layout(tt2);
    // (4,(2,2)):(2,(1,8))          <==  (i,(j,k))
    //        0    1    2    3      <== 1-D col coord
    //     (0,0) (1,0) (0,1) (1,1)  <== 2-D col coord (j,k)
    //     +----+----+----+----+
    //  0  |  0 |  1 |  8 |  9 |
    //     +----+----+----+----+
    //  1  |  2 |  3 | 10 | 11 |
    //     +----+----+----+----+
    //  2  |  4 |  5 | 12 | 13 |
    //     +----+----+----+----+
    //  3  |  6 |  7 | 14 | 15 |
    //     +----+----+----+----+
    // (i)
    print_latex(tt2);

    std::cout << rank(s8) << std::endl;
    std::cout << rank(tt) << std::endl;
    std::cout << rank(tt2) << std::endl;
    std::cout << depth(tt2) << std::endl;
    std::cout << shape(tt2) << std::endl;
    std::cout << stride(tt2) << std::endl;
    std::cout << size(tt2) << std::endl;
    std::cout << cosize(tt2) << std::endl;

    Layout tt3 = make_layout(make_shape(make_shape(2, 2), 2), make_stride(make_stride(4, 1), 2));
    print1D(tt3);
    //   0    4    1    5    2    6    3    7
    print_layout(tt3);
    //     ((2,2),2):((4,1),2) <==  ((i,j),k)
    //       0   1  <- (k)
    //     +---+---+
    //  0  | 0 | 2 | (0,0)
    //     +---+---+
    //  1  | 4 | 6 | (1,0)
    //     +---+---+
    //  2  | 1 | 3 | (0,1)
    //     +---+---+
    //  3  | 5 | 7 | (1,1)
    //     +---+---+
    //               (i,j)

    Layout tt4 = make_layout(make_shape(3, make_shape(2, 3)), make_stride(3, make_stride(12, 1)));
    print1D(tt4);
    //     0    3    6   12   15   18    1    4    7   13   16   19    2    5    8   14   17   20
    print_layout(tt4);
    // (3,(2,3)):(3,(12,1))
    //        0     1     2     3     4     5     <== 1-D col coord
    //      (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
    //     +-----+-----+-----+-----+-----+-----+
    //  0  |  0  |  12 |  1  |  13 |  2  |  14 |
    //     +-----+-----+-----+-----+-----+-----+
    //  1  |  3  |  15 |  4  |  16 |  5  |  17 |
    //     +-----+-----+-----+-----+-----+-----+
    //  2  |  6  |  18 |  7  |  19 |  8  |  20 |
    //     +-----+-----+-----+-----+-----+-----+
    return 0;
}