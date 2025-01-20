#include "CFInterpStencil.h"

template <>
const int CFInterpStencil<2, 1, 2, false, 2>::srcpos[5][2] = {{-1, 0},
                                                              {0, -1},
                                                              {0, 0},
                                                              {0, 1},
                                                              {1, 0}};

template <>
const int CFInterpStencil<2, 1, 2, false, 2>::dstpos[4][2] = {{0, 0},
                                                              {0, 1},
                                                              {1, 0},
                                                              {1, 1}};

template <>
const double CFInterpStencil<2, 1, 2, false, 2>::coef[4][5] = {
    {0.125, 0.125, 1, -0.125, -0.125},
    {0.125, -0.125, 1, 0.125, -0.125},
    {-0.125, 0.125, 1, -0.125, 0.125},
    {-0.125, -0.125, 1, 0.125, 0.125}};

template class CFInterpStencil<2, 1, 2, false, 2>;