#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::computeDivergence(const LD &u, LD &div) const {
  if (u.getnComps() != 2 || u.getCentering(0) != FaceCenter0 ||
      u.getCentering(1) != FaceCenter1) {
    throw std::runtime_error(
        "LevelOp:: Div(u) can only be applied on a face-centered vector.");
  }
  if (div.getnComps() != 1 || div.getCentering(0) != CellCenter) {
    throw std::runtime_error(
        "LevelOp:: The result of Div(u) should be a cell-centered scalar.");
  }
  auto dx = domain_.getDx();

  auto srcit = u.const_begin();
  for (auto dstit = div.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &u0 = srcit.getData()[0];
    const auto &u1 = srcit.getData()[1];
    auto box = dstit.getValidBox(0);
    auto &v = dstit.getData()[0];
    loop_box_2(box, i, j) {
      v(i, j) = (u0(i + 1, j) - u0(i, j)) / dx[0] +
                (u1(i, j + 1) - u1(i, j)) / dx[1];
    }
  }
}

template class LevelOp<2>;