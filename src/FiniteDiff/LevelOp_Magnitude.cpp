#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::computeMagnitude(const LD &u, LD &magu) const {
  if (u.getnComps() != 2 || u.getCentering(0) >= 0 || u.getCentering(1) >= 0 ||
      u.getCentering(0) != u.getCentering(1)) {
    throw std::runtime_error(
        "LevelOp:: computeMagnitude can only be "
        "applied on a cell-centered or node-centered vector.");
  }
  if (magu.getnComps() != 1 || magu.getCentering(0) != u.getCentering(0)) {
    throw std::runtime_error(
        "LevelOp:: the result of computeMagnitude should be a scalar with the "
        "same centering to the input vector.");
  }

  auto srcit = u.const_begin();
  for (auto dstit = magu.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &u0 = srcit.getData()[0];
    const auto &u1 = srcit.getData()[1];
    auto box = dstit.getValidBox(0);
    auto &v = dstit.getData()[0];
    loop_box_2(box, i, j) {
      v(i, j) = sqrt(u0(i, j) * u0(i, j) + u1(i, j) * u1(i, j));
    }
  }
}

template class LevelOp<2>;