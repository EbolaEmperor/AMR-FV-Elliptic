#include "FiniteDiff/AMRIntergridOp.h"

#include "AMRTools/Utilities.h"
#include "CFInterp/CFInterpStencil.h"
#include "Core/TensorExpr.h"
#include "Core/TensorSlice.h"

#include <cmath>
#include <cstring>

template <int Dim>
std::vector<Real> AMRIntergridOp<Dim>::computeNorm(const Vector<LD> &aDatas,
                                                   int q) const {
  UnitTimer::getInstance().begin("computeAMRNorm");
  assert(0 <= q && q <= 2);
  int nComps = aDatas[0].getnComps();
  Vector<LD> eDatas = aDatas;
  Vector<Real> sum(nComps, 0.);

  for (int i = amrHier_.size() - 1; i >= 0; --i) {
    auto lverr = lvOp_[i].computeNorm(eDatas[i], q);
    for (int comp = 0; comp < nComps; ++comp) {
      if (q == 0)
        sum[comp] = std::max(sum[comp], lverr[comp]);
      else
        sum[comp] += pow(lverr[comp], q);
    }
    if (i) {
      eDatas[i].memset(0);
      averageToCoarse(eDatas[i], eDatas[i - 1], i);
    }
  }
  if (q == 2) {
    for (int comp = 0; comp < nComps; ++comp)
      sum[comp] = sqrt(sum[comp]);
  }
  UnitTimer::getInstance().end("computeAMRNorm");
  return sum;
}

template <int Dim>
template <typename R>
void AMRIntergridOp<Dim>::checkValidation(const LevelData<R, Dim> &coarseData,
                                          const LevelData<R, Dim> &fineData,
                                          unsigned fineLevel) const {
#ifndef NDEBUG
  assert(fineLevel > 0 && fineLevel < amrHier_.size());
  assert(fineData.getMesh() == amrHier_.getMesh(fineLevel));
  assert(coarseData.getMesh() == amrHier_.getMesh(fineLevel - 1));
  assert(fineData.getnComps() == coarseData.getnComps());
  int nComps = fineData.getnComps();
  for (int i = 0; i < nComps; ++i)
    assert(fineData.getCentering(i) == coarseData.getCentering(i));
#endif
}

template <>
void AMRIntergridOp<2>::averageToCoarse(const LD &fineData,
                                        LD &coarseData,
                                        unsigned fineLevel) const {
  UnitTimer::getInstance().begin("averageToCoarse");
  checkValidation(coarseData, fineData, fineLevel);
  int nComps = fineData.getnComps();

  // Get essential infomations
  const auto &parent = amrHier_.getParents(fineLevel);
  int ratio = amrHier_.getRefRatioToNextLevel(fineLevel - 1);
  double w = 1. / (ratio * ratio);

  // Loop the box-datas
  for (auto it = fineData.const_begin(); it.ok(); ++it) {
    int box_id = it.getBoxID();
    int parent_box_id = parent[box_id];

    for (int comp = 0; comp < nComps; ++comp) {
      int cent = fineData.getCentering(comp);
      const auto &fine_box_data = it.getData()[comp];
      auto &coarse_box_data = coarseData.getBoxData(parent_box_id)[comp];
      Box<2> valid_box = it.getValidBox(comp);
      Box<2> valid_coarse_box = valid_box.getCoarsened(ratio, cent);

      // clear the coarse box-data where the fine box covers.
      coarse_box_data.slice(valid_coarse_box) = 0.;

      // take averages.
      if (cent == CellCenter) {
#pragma omp parallel for default(shared) schedule(static)
        loop_box_2(valid_coarse_box, i0, i1) {
          for (int i = 0; i < ratio; ++i)
            for (int j = 0; j < ratio; ++j)
              coarse_box_data(i0, i1) +=
                  fine_box_data(i0 * ratio + i, i1 * ratio + j) * w;
        }
      } else if (cent == NodeCenter) {
#pragma omp parallel for default(shared) schedule(static)
        loop_box_2(valid_coarse_box, i0, i1) {
          coarse_box_data(i0, i1) = fine_box_data(i0 * ratio, i1 * ratio);
        }
      } else if (cent == FaceCenter0) {
#pragma omp parallel for default(shared) schedule(static)
        loop_box_2(valid_coarse_box, i0, i1) {
          for (int j = 0; j < ratio; j++)
            coarse_box_data(i0, i1) +=
                fine_box_data(i0 * ratio, i1 * ratio + j) / ratio;
        }
      } else {  // FaceCenter1
#pragma omp parallel for default(shared) schedule(static)
        loop_box_2(valid_coarse_box, i0, i1) {
          for (int j = 0; j < ratio; j++)
            coarse_box_data(i0, i1) +=
                fine_box_data(i0 * ratio + j, i1 * ratio) / ratio;
        }
      }
    }
  }
  UnitTimer::getInstance().end("averageToCoarse");
}

template <>
template <typename R>
void AMRIntergridOp<2>::constantInterpolateIncr(
    const LevelData<R, 2> &coarseData,
    LevelData<R, 2> &fineData,
    unsigned fineLevel) const {
  UnitTimer::getInstance().begin("constantInterp");
  checkValidation(coarseData, fineData, fineLevel);
  int nComps = fineData.getnComps();

  // Get essential infomationscDatas[domainID.value()][face]
  const auto &parent = amrHier_.getParents(fineLevel);
  int ratio = amrHier_.getRefRatioToNextLevel(fineLevel - 1);
  double w = 1. / (ratio * ratio);

  // Loop the box-datas
  for (auto it = fineData.begin(); it.ok(); ++it) {
    int box_id = it.getBoxID();
    int parent_box_id = parent[box_id];

    for (int comp = 0; comp < nComps; ++comp) {
      int cent = fineData.getCentering(comp);
      auto &fine_box_data = it.getData()[comp];
      const auto &coarse_box_data = coarseData.getBoxData(parent_box_id)[comp];
      Box<2> valid_box = it.getValidBox(comp);
      Box<2> valid_coarse_box = valid_box.getCoarsened(ratio, cent);

      // piecewise constantly interpolation.
      if (cent == CellCenter) {
        loop_box_2(valid_box, i0, i1) {
          fine_box_data(i0, i1) += coarse_box_data(i0 / ratio, i1 / ratio);
        }
      } else if (cent == NodeCenter) {
        loop_box_2(valid_coarse_box, i0, i1) {
          fine_box_data(i0 * ratio, i1 * ratio) += coarse_box_data(i0, i1);
        }
      } else if (cent == FaceCenter0) {
        loop_box_2(valid_coarse_box, i0, i1) {
          for (int j = 0; j < ratio; j++)
            fine_box_data(i0 * ratio, i1 * ratio + j) +=
                coarse_box_data(i0, i1);
        }
      } else {  // FaceCenter1
        loop_box_2(valid_coarse_box, i0, i1) {
          for (int j = 0; j < ratio; j++)
            fine_box_data(i0 * ratio + j, i1 * ratio) +=
                coarse_box_data(i0, i1);
        }
      }
    }
  }
  UnitTimer::getInstance().end("constantInterp");
}

template <>
template <int Order, int Addition>
void AMRIntergridOp<2>::interpolateIncr(const LD &coarseData,
                                        LD &fineData,
                                        unsigned fineLevel) const {
  UnitTimer::getInstance().begin("highOrderInterp");
  checkValidation(coarseData, fineData, fineLevel);
  int nComps = fineData.getnComps();

  // Get essential infomationscDatas[domainID.value()][face]
  const auto &parent = amrHier_.getParents(fineLevel);
  int ratio = amrHier_.getRefRatioToNextLevel(fineLevel - 1);
  double w = 1. / (ratio * ratio);

  const int N = CFInterpStencilConst<2, Order, 2, false, Addition>::nSrc;
  const int(*srcpos)[2];
  const int(*dstpos)[2];
  const double(*coef)[N];

  if (ratio == 2) {
    CFInterpStencil<2, Order, 2, false, Addition>::getArr(
        srcpos, dstpos, coef);
  } else if (ratio == 4) {
    CFInterpStencil<2, Order, 4, false, Addition>::getArr(
        srcpos, dstpos, coef);
  } else {
    throw std::runtime_error("Invalid ref_ratio, Only supports 2 and 4.");
  }

  // Loop the box-datas
  for (auto it = fineData.begin(); it.ok(); ++it) {
    int box_id = it.getBoxID();
    int parent_box_id = parent[box_id];

    for (int comp = 0; comp < nComps; ++comp) {
      int cent = fineData.getCentering(comp);
      auto &fine_box_data = it.getData()[comp];
      const auto &coarse_box_data = coarseData.getBoxData(parent_box_id)[comp];
      Box<2> valid_box = it.getValidBox(comp);
      Box<2> valid_coarse_box = valid_box.getCoarsened(ratio, cent);

      // piecewise quarticly interpolation.
      if (cent == CellCenter) {
        int tot = ratio * ratio;
        loop_box_2(valid_coarse_box, i0, i1) {
          for (int d = 0; d < tot; ++d) {
            Real tmp = 0.;
            for (int c = 0; c < N; ++c)
              tmp += coef[d][c] *
                     coarse_box_data(i0 + srcpos[c][0], i1 + srcpos[c][1]);
            fine_box_data(ratio * i0 + dstpos[d][0],
                          ratio * i1 + dstpos[d][1]) += tmp;
          }
        }
      } else {
        throw std::runtime_error(
            "interpolatIncr only support cell-center now.");
      }
    }
  }
  UnitTimer::getInstance().end("highOrderInterp");
}

template <>
void AMRIntergridOp<2>::autoInterpolateIncr(const LD &coarseData,
                                            LD &fineData,
                                            unsigned fineLevel,
                                            int order) const {
  if (order == 0)
    constantInterpolateIncr(coarseData, fineData, fineLevel);
  else if (order == 1)
    interpolateIncr<1, 2>(coarseData, fineData, fineLevel);
  else if (order == 2)
    interpolateIncr<2, 3>(coarseData, fineData, fineLevel);
  else if (order == 4)
    interpolateIncr<4, 6>(coarseData, fineData, fineLevel);
  else {
    throw std::runtime_error(
        "AMRIntergridOp:: unsupported interpolation order.");
  }
}

template <>
void AMRIntergridOp<2>::fillCorner(LD &aData, unsigned level) const {
  gstFiller_[level].fillCorners(aData);
}

template <>
void AMRIntergridOp<2>::interpolateToFineGhost(const LD &coarseData,
                                               LD &fineData,
                                               unsigned fineLevel) const {
  UnitTimer::getInstance().begin("CF-GhostInterp");
  checkValidation(coarseData, fineData, fineLevel);
  const int nComps = fineData.getnComps();
  const int nG = amrHier_.getnGhost();
  const int ratio = amrHier_.getRefRatioToNextLevel(fineLevel - 1);

  // Only support cell-center now...
  for (int comp = 0; comp < nComps; ++comp)
    assert(fineData.getCentering(comp) == CellCenter);

  const int addition = 6;
  const int N = CFInterpStencilConst<2, 4, 2, true, addition>::nSrc;
  const int(*srcpos)[2];
  const int(*dstpos)[2];
  const double(*coef)[N];

  if (ratio == 2) {
    CFInterpStencil<2, 4, 2, true, addition>::getArr(srcpos, dstpos, coef);
  } else if (ratio == 4) {
    CFInterpStencil<2, 4, 4, true, addition>::getArr(srcpos, dstpos, coef);
  } else {
    throw std::runtime_error("Invalid ref_ratio, Only supports 2 and 4.");
  }

  const auto &lgb = amrHier_.getLevelGhostBoxes(fineLevel);
  for (auto it = lgb.begin(); it.ok(); ++it) {
    if (lgb.getProcID(it) != ProcID(MPI_COMM_WORLD))
      continue;

    auto face = lgb.getBdryOfWhichFace(it);
    if (!face.has_value() || lgb.getBdryOfWhichDomain(it).has_value())
      continue;

    for (int comp = 0; comp < nComps; ++comp) {
      int box_id = lgb.getBelongs(it);
      auto &aData = fineData.getBoxData(box_id)[comp];
      int parent_box_id = amrHier_.getParent(fineLevel, box_id);
      const auto &pData = coarseData.getBoxData(parent_box_id)[comp];

      if (face.value() < 2) {
        // left-face or right-face
        int i0 = it->lo()[0] / ratio;
        bool lrinv = face.value() & 1;
        int fglr = lrinv ? -1 : 1;
        for (int i1 = it->lo()[1] / ratio; i1 <= it->hi()[1] / ratio; ++i1) {
          for (int k = 0; k < nG * ratio; k++) {
            int finei0 = lrinv ? (i0 + 1) * ratio - 1 - dstpos[k][0]
                               : i0 * ratio + dstpos[k][0];
            aData(finei0, i1 * ratio + dstpos[k][1]) = 0;
            for (int j = 0; j < N; j++)
              aData(finei0, i1 * ratio + dstpos[k][1]) +=
                  coef[k][j] *
                  pData(i0 + srcpos[j][0] * fglr, i1 + srcpos[j][1]);
          }
        }
      } else {
        // down-face or up-face
        int i1 = it->lo()[1] / ratio;
        bool duinv = face.value() & 1;
        int fgdu = duinv ? -1 : 1;
        for (int i0 = it->lo()[0] / ratio; i0 <= it->hi()[0] / ratio; ++i0) {
          for (int k = 0; k < nG * ratio; k++) {
            int finei1 = duinv ? (i1 + 1) * ratio - 1 - dstpos[k][0]
                               : i1 * ratio + dstpos[k][0];
            aData(i0 * ratio + dstpos[k][1], finei1) = 0;
            for (int j = 0; j < N; j++)
              aData(i0 * ratio + dstpos[k][1], finei1) +=
                  coef[k][j] *
                  pData(i0 + srcpos[j][1], i1 + srcpos[j][0] * fgdu);
          }
        }
      }
    }
  }
  fillCorner(fineData, fineLevel);
  UnitTimer::getInstance().end("CF-GhostInterp");
}

template <int Dim>
void AMRIntergridOp<Dim>::fillGhosts(
    Vector<LD> &phi,
    const Vector<Vector<Array<Tensor<Real, Dim - 1>, 2 * Dim>>> &cDatas,
    const Vector<Array<char, 2 * Dim>> &bcType) const {
  averageToCoarse(phi);
  exchangeInLevel(phi);

  for (unsigned i = 0; i < amrHier_.size(); i++) {
    int nComps = phi[i].getnComps();
    for (int comp = 0; comp < nComps; ++comp)
      gstFiller_[i].fillGhosts(phi[i], cDatas[i], bcType, comp);

    if (i + 1 < amrHier_.size())
      interpolateToFineGhost(phi[i], phi[i + 1], i + 1);
  }
}

template class AMRIntergridOp<2>;

template void AMRIntergridOp<2>::constantInterpolateIncr(
    const LD &coarseData,
    LD &fineData,
    unsigned fineLevel) const;

template void AMRIntergridOp<2>::constantInterpolateIncr(
    const LevelData<int, 2> &coarseData,
    LevelData<int, 2> &fineData,
    unsigned fineLevel) const;

template void AMRIntergridOp<2>::constantInterpolateIncr(
    const LevelData<bool, 2> &coarseData,
    LevelData<bool, 2> &fineData,
    unsigned fineLevel) const;