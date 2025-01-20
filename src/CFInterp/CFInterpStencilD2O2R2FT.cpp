#include "CFInterpStencil.h"

template <>
const int CFInterpStencil<2, 2, 2, false, 3>::srcpos[9][2] = {
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 0},
    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1},
};

template <>
const int CFInterpStencil<2, 2, 2, false, 3>::dstpos[4][2] = {{0, 0},
                                                              {0, 1},
                                                              {1, 0},
                                                              {1, 1}};

template <>
const double CFInterpStencil<2, 2, 2, false, 3>::coef[4][9] = {
    {1. / 64,
     1. / 8,
     -1. / 64,
     1. / 8,
     1,
     -1. / 8,
     -1. / 64,
     -1. / 8,
     1. / 64},
    {-1. / 64,
     1. / 8,
     1. / 64,
     -1. / 8,
     1,
     1. / 8,
     1. / 64,
     -1. / 8,
     -1. / 64},
    {-1. / 64,
     -1. / 8,
     1. / 64,
     1. / 8,
     1,
     -1. / 8,
     1. / 64,
     1. / 8,
     -1. / 64},
    {1. / 64,
     -1. / 8,
     -1. / 64,
     -1. / 8,
     1,
     1. / 8,
     -1. / 64,
     1. / 8,
     1. / 64},
};

template class CFInterpStencil<2, 2, 2, false, 3>;