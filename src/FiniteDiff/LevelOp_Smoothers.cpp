#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::relaxJacobi(const LD &phi,
                             const LD &rhs,
                             LD &smoothed,
                             Real alpha,
                             Real beta,
                             Real jW) const {
  UnitTimer::getInstance().begin("relaxJacobi");
  int nComps = rhs.getnComps();
  auto dx = domain_.getDx();

  Real w0 = -beta * 1. / (12 * dx[0] * dx[0]);
  Real w1 = -beta * 1. / (12 * dx[1] * dx[1]);
  Real w = jW / (alpha + 30 * w0 + 30 * w1);

  auto rhsit = rhs.const_begin();
  auto smtit = smoothed.begin();
  auto phiit = phi.const_begin();
  for (; phiit.ok(); ++rhsit, ++phiit, ++smtit) {
    for (int comp = 0; comp < nComps; ++comp) {
      auto box = rhsit.getValidBox(comp);
      const auto &phiData = phiit.getData()[comp];
      const auto &rhsData = rhsit.getData()[comp];
      auto &smtData = smtit.getData()[comp];
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i, j) {
        smtData(i, j) =
            (1 - jW) * phiData(i, j) +
            w * (rhsData(i, j) +
                 w0 * (-phiData(i - 2, j) + 16 * phiData(i - 1, j) +
                       16 * phiData(i + 1, j) - phiData(i + 2, j)) +
                 w1 * (-phiData(i, j - 2) + 16 * phiData(i, j - 1) +
                       16 * phiData(i, j + 1) - phiData(i, j + 2)));
      }
    }
  }
  UnitTimer::getInstance().end("relaxJacobi");
}

template <>
void LevelOp<2>::relaxJacobiOd2(const LD &phi,
                                const LD &rhs,
                                LD &smoothed,
                                Real alpha,
                                Real beta,
                                Real jW) const {
  UnitTimer::getInstance().begin("relaxJacobi");
  int nComps = rhs.getnComps();
  auto dx = domain_.getDx();

  Real w0 = -beta * 1. / (dx[0] * dx[0]);
  Real w1 = -beta * 1. / (dx[1] * dx[1]);
  Real w = jW / (alpha + 2 * w0 + 2 * w1);

  auto rhsit = rhs.const_begin();
  auto smtit = smoothed.begin();
  auto phiit = phi.const_begin();
  for (; phiit.ok(); ++rhsit, ++phiit, ++smtit) {
    for (int comp = 0; comp < nComps; ++comp) {
      auto box = rhsit.getValidBox(comp);
      const auto &phiData = phiit.getData()[comp];
      const auto &rhsData = rhsit.getData()[comp];
      auto &smtData = smtit.getData()[comp];
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i, j) {
        smtData(i, j) =
            (1 - jW) * phiData(i, j) +
            w * (rhsData(i, j) + w0 * (phiData(i - 1, j) + phiData(i + 1, j)) +
                 w1 * (phiData(i, j - 1) + phiData(i, j + 1)));
      }
    }
  }
  UnitTimer::getInstance().end("relaxJacobi");
}

template <>
void LevelOp<2>::relaxSOROd2(const LD &phi,
                             const LD &rhs,
                             LD &smoothed,
                             Real alpha,
                             Real beta,
                             Real jW) const {
  UnitTimer::getInstance().begin("relaxSOR");
  int nComps = rhs.getnComps();
  auto dx = domain_.getDx();

  Real w0 = -beta * 1. / (dx[0] * dx[0]);
  Real w1 = -beta * 1. / (dx[1] * dx[1]);
  Real w = 1. / (alpha + 2 * w0 + 2 * w1);

  smoothed = phi;
  auto rhsit = rhs.const_begin();
  auto smtit = smoothed.begin();
  auto phiit = phi.const_begin();
  for (; phiit.ok(); ++rhsit, ++phiit, ++smtit) {
    for (int comp = 0; comp < nComps; ++comp) {
      auto box = rhsit.getValidBox(comp);
      const auto &phiData = phiit.getData()[comp];
      const auto &rhsData = rhsit.getData()[comp];
      auto &smtData = smtit.getData()[comp];

      // Use red-black ordering here.
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i, j) {
        if ((i + j) & 1)
          smtData(i, j) = w * (rhsData(i, j) +
                               w0 * (smtData(i - 1, j) + smtData(i + 1, j)) +
                               w1 * (smtData(i, j - 1) + smtData(i, j + 1)));
      }
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i, j) {
        if (!((i + j) & 1))
          smtData(i, j) = w * (rhsData(i, j) +
                               w0 * (smtData(i - 1, j) + smtData(i + 1, j)) +
                               w1 * (smtData(i, j - 1) + smtData(i, j + 1)));
      }
    }
  }
  smoothed = smoothed * jW + phi * (1 - jW);
  UnitTimer::getInstance().end("relaxSOR");
}

template class LevelOp<2>;