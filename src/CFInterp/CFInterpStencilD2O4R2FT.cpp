#include "CFInterpStencil.h"

template <>
const int CFInterpStencil<2, 4, 2, false, 6>::srcpos[21][2] = {
    {-2, 0}, {-1, -1}, {-1, 0},  {-1, 1}, {0, -2}, {0, -1}, {0, 0},
    {0, 1},  {0, 2},   {1, -1},  {1, 0},  {1, 1},  {2, 0},  {-2, 1},
    {-1, 2}, {-2, -1}, {-1, -2}, {1, 2},  {2, 1},  {1, -2}, {2, -1}};

template <>
const int CFInterpStencil<2, 4, 2, false, 6>::dstpos[4][2] = {{0, 0},
                                                              {0, 1},
                                                              {1, 0},
                                                              {1, 1}};

template <>
const double CFInterpStencil<2, 4, 2, false, 6>::coef[4][21] = {
    {-3. / 128,  7. / 256,  11. / 64,  -7. / 256,  -3. / 128,  11. / 64,
     1,          -11. / 64, 3. / 128,  -7. / 256,  -11. / 64,  7. / 256,
     3. / 128,   3. / 1024, 3. / 1024, -3. / 1024, -3. / 1024, -3. / 1024,
     -3. / 1024, 3. / 1024, 3. / 1024},
    {-3. / 128, -7. / 256,  11. / 64,   7. / 256,  3. / 128,  -11. / 64,
     1,         11. / 64,   -3. / 128,  7. / 256,  -11. / 64, -7. / 256,
     3. / 128,  -3. / 1024, -3. / 1024, 3. / 1024, 3. / 1024, 3. / 1024,
     3. / 1024, -3. / 1024, -3. / 1024},
    {3. / 128,  -7. / 256,  -11. / 64,  7. / 256,  -3. / 128, 11. / 64,
     1,         -11. / 64,  3. / 128,   7. / 256,  11. / 64,  -7. / 256,
     -3. / 128, -3. / 1024, -3. / 1024, 3. / 1024, 3. / 1024, 3. / 1024,
     3. / 1024, -3. / 1024, -3. / 1024},
    {3. / 128,   7. / 256,  -11. / 64, -7. / 256,  3. / 128,   -11. / 64,
     1,          11. / 64,  -3. / 128, -7. / 256,  11. / 64,   7. / 256,
     -3. / 128,  3. / 1024, 3. / 1024, -3. / 1024, -3. / 1024, -3. / 1024,
     -3. / 1024, 3. / 1024, 3. / 1024}};

template class CFInterpStencil<2, 4, 2, false, 6>;