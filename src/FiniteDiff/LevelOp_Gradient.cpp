#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::computeGradientCell2Face(const LD &phi, LD &grad) const {
  auto dx = domain_.getDx();
  const Real w[2] = {1. / (12. * dx[0]), 1. / (12. * dx[1])};

  auto srcit = phi.const_begin();
  for (auto dstit = grad.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &srcData = srcit.getData()[0];
    auto box0 = dstit.getValidBox(0);
    auto &dstData0 = dstit.getData()[0];
    loop_box_2(box0, i, j) {
      dstData0(i, j) = w[0] * (srcData(i - 2, j) - 15. * srcData(i - 1, j) +
                               15. * srcData(i, j) - srcData(i + 1, j));
    }
    auto box1 = dstit.getValidBox(1);
    auto &dstData1 = dstit.getData()[1];
    loop_box_2(box1, i, j) {
      dstData1(i, j) = w[1] * (srcData(i, j - 2) - 15. * srcData(i, j - 1) +
                               15. * srcData(i, j) - srcData(i, j + 1));
    }
  }
}

template <>
void LevelOp<2>::computeGradientCell2Cell(const LD &phi, LD &grad) const {
  auto dx = domain_.getDx();
  const Real w[2] = {1. / (12. * dx[0]), 1. / (12. * dx[1])};

  auto srcit = phi.const_begin();
  for (auto dstit = grad.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &srcData = srcit.getData()[0];
    auto box0 = dstit.getValidBox(0);
    auto &dstData0 = dstit.getData()[0];
    loop_box_2(box0, i, j) {
      dstData0(i, j) = w[0] * (srcData(i - 2, j) - 8. * srcData(i - 1, j) +
                               8. * srcData(i + 1, j) - srcData(i + 2, j));
    }
    auto box1 = dstit.getValidBox(1);
    auto &dstData1 = dstit.getData()[1];
    loop_box_2(box1, i, j) {
      dstData1(i, j) = w[1] * (srcData(i, j - 2) - 8. * srcData(i, j - 1) +
                               8. * srcData(i, j + 1) - srcData(i, j + 2));
    }
  }
}

template <int Dim>
void LevelOp<Dim>::computeGradient(const LD &phi, LD &grad) const {
  if (phi.getnComps() != 1 || phi.getCentering(0) != CellCenter) {
    throw std::runtime_error(
        "LevelOp:: Grad(phi) can only be applied on a cell-centered scalar.");
  }

  if (grad.getnComps() == Dim) {
    bool isFaceCenteredVector = true;
    for (int comp = 0; comp < Dim; ++comp)
      if (grad.getCentering(comp) != comp)
        isFaceCenteredVector = false;
    if (isFaceCenteredVector) {
      computeGradientCell2Face(phi, grad);
      return;
    }

    bool isCellCenteredVector = true;
    for (int comp = 0; comp < Dim; ++comp)
      if (grad.getCentering(comp) != CellCenter)
        isCellCenteredVector = false;
    if (isCellCenteredVector) {
      computeGradientCell2Cell(phi, grad);
      return;
    }
  }

  throw std::runtime_error(
      "LevelOp:: The result of Grad(phi) should be a face-centered vector\n"
      " or a cell-centered vector");
}

template class LevelOp<2>;