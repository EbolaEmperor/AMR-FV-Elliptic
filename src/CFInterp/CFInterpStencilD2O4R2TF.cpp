#include "CFInterpStencil.h"

template <>
const int CFInterpStencil<2, 4, 2, true, 0>::srcpos[15][2] = {{-1, -1},
                                                              {-1, 0},
                                                              {-1, 1},
                                                              {0, -2},
                                                              {0, -1},
                                                              {0, 0},
                                                              {0, 1},
                                                              {0, 2},
                                                              {1, -1},
                                                              {1, 0},
                                                              {1, 1},
                                                              {1, 2},
                                                              {2, 0},
                                                              {2, 1},
                                                              {3, 0}};

template <>
const int CFInterpStencil<2, 4, 2, true, 0>::dstpos[4][2] = {{0, 0},
                                                             {0, 1},
                                                             {1, 0},
                                                             {1, 1}};

template <>
const double CFInterpStencil<2, 4, 2, true, 0>::coef[4][15] = {{1. / 64,
                                                                11. / 256,
                                                                -1. / 256,
                                                                -3. / 128,
                                                                41. / 256,
                                                                167. / 128,
                                                                -31. / 128,
                                                                9. / 256,
                                                                -1. / 256,
                                                                -61. / 128,
                                                                11. / 128,
                                                                -3. / 256,
                                                                39. / 256,
                                                                -3. / 256,
                                                                -3. / 128},

                                                               {-1. / 64,
                                                                17. / 256,
                                                                1. / 256,
                                                                3. / 128,
                                                                -41. / 256,
                                                                149. / 128,
                                                                31. / 128,
                                                                -9. / 256,
                                                                1. / 256,
                                                                -43. / 128,
                                                                -11. / 128,
                                                                3. / 256,
                                                                33. / 256,
                                                                3. / 256,
                                                                -3. / 128},

                                                               {-1. / 64,
                                                                -11. / 256,
                                                                1. / 256,
                                                                -3. / 128,
                                                                47. / 256,
                                                                89. / 128,
                                                                -13. / 128,
                                                                3. / 256,
                                                                1. / 256,
                                                                61. / 128,
                                                                -11. / 128,
                                                                3. / 256,
                                                                -39. / 256,
                                                                3. / 256,
                                                                3. / 128},

                                                               {1. / 64,
                                                                -17. / 256,
                                                                -1. / 256,
                                                                3. / 128,
                                                                -47. / 256,
                                                                107. / 128,
                                                                13. / 128,
                                                                -3. / 256,
                                                                -1. / 256,
                                                                43. / 128,
                                                                11. / 128,
                                                                -3. / 256,
                                                                -33. / 256,
                                                                -3. / 256,
                                                                3. / 128}};

template class CFInterpStencil<2, 4, 2, true, 0>;