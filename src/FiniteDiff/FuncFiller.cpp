#include "FiniteDiff/FuncFiller.h"

template class FuncFiller<2>;

typedef double (*RealFunc2D)(const Vec<Real, 2> &);

template void FuncFiller<2>::fillAvr(LevelData<Real, 2> &,
                                     const RealFunc2D &,
                                     bool,
                                     unsigned) const;

template void FuncFiller<2>::fillBdryAvr(Vector<Array<Tensor<Real, 1>, 4>> &,
                                         const RealFunc2D &,
                                         int,
                                         int,
                                         int) const;

template void FuncFiller<2>::fillBdryAvr(Vector<Array<Tensor<Real, 1>, 4>> &,
                                         const RealFunc2D &,
                                         int) const;