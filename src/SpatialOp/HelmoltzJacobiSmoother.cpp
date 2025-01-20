#include "SpatialOp/HelmoltzJacobiSmoother.h"

#include "AMRTools/LevelDataExpr.h"

template <>
void HelmoltzJacobiSmoother<2, 4>::apply(LD &dst, const LD &src) const {
  lvOp_.computeHelmoltz(src, dst, alpha_, beta_);
}

template <>
void HelmoltzJacobiSmoother<2, 4>::relax(const LD &phi,
                                         const LD &rhs,
                                         LD &smoothed) {
  lvOp_.relaxJacobi(phi, rhs, smoothed, alpha_, beta_, jacobiWeight_);
}

template <>
void HelmoltzJacobiSmoother<2, 2>::apply(LD &dst, const LD &src) const {
  lvOp_.computeHelmoltzOd2(src, dst, alpha_, beta_);
}

template <>
void HelmoltzJacobiSmoother<2, 2>::relax(const LD &phi,
                                         const LD &rhs,
                                         LD &smoothed) {
  lvOp_.relaxSOROd2(phi, rhs, smoothed, alpha_, beta_, jacobiWeight_);
}

template class HelmoltzJacobiSmoother<2, 4>;
template class HelmoltzJacobiSmoother<2, 2>;