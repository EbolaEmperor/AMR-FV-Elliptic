#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::computeLaplacian(const LD &src, LD &dst) const {
  UnitTimer::getInstance().begin("computeLaplacian");
  int nComps = src.getnComps();
  auto dx = domain_.getDx();
  Real w0 = 1. / (12 * dx[0] * dx[0]);
  Real w1 = 1. / (12 * dx[1] * dx[1]);

  auto srcit = src.const_begin();
  for (auto dstit = dst.begin(); dstit.ok(); ++dstit, ++srcit) {
    for (int comp = 0; comp < nComps; ++comp) {
      auto box = dstit.getValidBox(comp);
      auto &dstData = dstit.getData()[comp];
      const auto &srcData = srcit.getData()[comp];
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i, j) {
        dstData(i, j) = w0 * (-srcData(i - 2, j) + 16 * srcData(i - 1, j) -
                              30 * srcData(i, j) + 16 * srcData(i + 1, j) -
                              srcData(i + 2, j)) +
                        w1 * (-srcData(i, j - 2) + 16 * srcData(i, j - 1) -
                              30 * srcData(i, j) + 16 * srcData(i, j + 1) -
                              srcData(i, j + 2));
      }
    }
  }
  UnitTimer::getInstance().end("computeLaplacian");
}

template <>
void LevelOp<2>::computeD2(const LD &phi, LD &d2phi) const {
  const auto dx = domain_.getDx();
  const Real w0 = 1. / (12 * dx[0] * dx[0]);
  const Real w1 = 1. / (12 * dx[1] * dx[1]);

  auto srcit = phi.const_begin();
  for (auto dstit = d2phi.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &srcData = srcit.getData()[0];
    auto box0 = dstit.getValidBox(0);
    auto &dstData0 = dstit.getData()[0];
    loop_box_2(box0, i, j) {
      dstData0(i, j) = w0 * (-srcData(i - 2, j) + 16. * srcData(i - 1, j) -
                             30. * srcData(i, j) + 16. * srcData(i + 1, j) -
                             srcData(i + 2, j));
    }
    auto box1 = dstit.getValidBox(1);
    auto &dstData1 = dstit.getData()[1];
    loop_box_2(box1, i, j) {
      dstData1(i, j) = w1 * (-srcData(i, j - 2) + 16. * srcData(i, j - 1) -
                             30. * srcData(i, j) + 16. * srcData(i, j + 1) -
                             srcData(i, j + 2));
    }
  }
}

template <int Dim>
void LevelOp<Dim>::computeLaplacianOd2(const LD &src, LD &dst) const {
  UnitTimer::getInstance().begin("computeLaplacian");
  int nComps = src.getnComps();
  auto dx = domain_.getDx();
  Real w0 = 1. / (dx[0] * dx[0]);
  Real w1 = 1. / (dx[1] * dx[1]);

  auto srcit = src.const_begin();
  for (auto dstit = dst.begin(); dstit.ok(); ++dstit, ++srcit) {
    for (int comp = 0; comp < nComps; ++comp) {
      auto box = dstit.getValidBox(comp);
      auto &dstData = dstit.getData()[comp];
      const auto &srcData = srcit.getData()[comp];
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i, j) {
        dstData(i, j) =
            w0 * (srcData(i - 1, j) - 2 * srcData(i, j) + srcData(i + 1, j)) +
            w1 * (srcData(i, j - 1) - 2 * srcData(i, j) + srcData(i, j + 1));
      }
    }
  }
  UnitTimer::getInstance().end("computeLaplacian");
}

template <int Dim>
void LevelOp<Dim>::computeHelmoltz(const LD &phi,
                                   LD &helm,
                                   Real alpha,
                                   Real beta) const {
  computeLaplacian(phi, helm);
  if (alpha == 0) {
    if (beta != 1)
      helm = helm * beta;
  } else {
    helm = phi * alpha + helm * beta;
  }
}

template <int Dim>
void LevelOp<Dim>::computeHelmoltzOd2(const LD &phi,
                                      LD &helm,
                                      Real alpha,
                                      Real beta) const {
  computeLaplacianOd2(phi, helm);
  if (alpha == 0) {
    if (beta != 1)
      helm = helm * beta;
  } else {
    helm = phi * alpha + helm * beta;
  }
}

template class LevelOp<2>;