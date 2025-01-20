#include <AMRTools/LevelDataExpr.h>
#include <SpatialOp/LinearSolver.h>

template <int Dim>
void LinearSolver<LevelData<Real, Dim>>::computeResidual(
    LevelData<Real, Dim> &dst,
    const LevelData<Real, Dim> &src,
    const LevelData<Real, Dim> &rhs) const {
  apply(dst, src);
  dst = -dst + rhs;
  // TODO: can be deleted
  dst.exchangeAll();
  // assert(dst.getMesh() == src.getMesh());
  // assert(src.getMesh() == rhs.getMesh());
  // assert(dst.getnComps() == src.getnComps());
  // assert(src.getnComps() == rhs.getnComps());
  // auto dst_itr = dst.begin();
  // auto src_itr = src.const_begin();
  // auto rhs_itr = rhs.const_begin();
  // for (; dst_itr.ok(); ++dst_itr, ++src_itr, ++rhs_itr) {
  //   auto &dst_data = dst_itr.getData();
  //   const auto &src_data = src_itr.getData();
  //   const auto &rhs_data = rhs_itr.getData();
  // }
};

template class LinearSolver<LevelData<Real, 2>>;