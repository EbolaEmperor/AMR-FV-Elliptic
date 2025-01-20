#include "SpatialOp/AMRMultigrid.h"

#include "AMRTools/LevelDataExpr.h"
#include "SpatialOp/AMRMGLevelOp.h"
#include "SpatialOp/HelmoltzJacobiSmoother.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

template <int Dim>
void AMRMultigrid<Dim>::setParam(int preSmooth,
                                 int postSmooth,
                                 int bottomSmooth,
                                 int maxIter,
                                 Real relTol,
                                 Real stallTol,
                                 bool useFMG,
                                 int FMGInterpOd) {
  if (!defined_) {
    throw std::runtime_error(
        "AMRMultigrid:: please call setParam() after defined.");
  }
  maxIter_ = maxIter;
  relTol_ = relTol;
  stallTol_ = stallTol;
  useFMG_ = useFMG;
  if (useFMG) {
    if (FMGInterpOd < 0 || FMGInterpOd > 4 || FMGInterpOd == 3)
      throw std::runtime_error("AMRMultigrid:: unsupported FMGInterpOd.");
    FMGInterpOd_ = FMGInterpOd;
  }
  for (unsigned i = 1; i < amrHier_.size(); ++i) {
    mgSmoothers_[i]->setParam(preSmooth, postSmooth);
  }
  mgSmoothers_[0]->setParam(preSmooth, postSmooth, bottomSmooth, false);

  // Still in experiment...
  // mgSmoothers_[0]->setParam(preSmooth, postSmooth, bottomSmooth, useFMG ^
  // 1);
}

template <int Dim>
void AMRMultigrid<Dim>::define(Real alpha,
                               Real beta,
                               int centering,
                               int bottomNumLevel,
                               Real jW,
                               std::shared_ptr<LS> bottomSolver) {
  Vector<int> cent(1, centering);
  define(alpha, beta, 1, cent, bottomNumLevel, jW, bottomSolver);
}

template <int Dim>
void AMRMultigrid<Dim>::define(Real alpha,
                               Real beta,
                               unsigned nComps,
                               const Vector<int> &centering,
                               int bottomNumLevel,
                               Real jW,
                               std::shared_ptr<LS> bottomSolver) {
  if (defined_) {
    throw std::runtime_error("AMRMultigrid:: redefined.");
  }
  defined_ = true;

  UnitTimer::getInstance().begin("initAMRMultigrid");
  // Only support Cell-Center now...
  assert(centering.size() == nComps);
  for (unsigned comp = 0; comp < nComps; ++comp)
    assert(centering[comp] == CellCenter);

  nComps_ = nComps;
  centering_ = centering;

  amrIntergridOp_ =
      std::make_unique<AMRIntergridOp<Dim>>(AMRIntergridOp<Dim>(amrHier_));

  mgSmoothers_.resize(amrHier_.size());
  for (unsigned i = 0; i < amrHier_.size(); ++i) {
    mgSmoothers_[i] = std::make_unique<AMRMGLevelOp<Dim>>(
        AMRMGLevelOp<Dim>(amrHier_.getDomain(i)));
    int nLevel =
        i ? (int)log2(amrHier_.getRefRatioToNextLevel(i - 1)) : bottomNumLevel;
    std::shared_ptr<LS> bs = i ? nullptr : bottomSolver;
    mgSmoothers_[i]->define(
        amrHier_.getMesh(i), alpha, beta, nLevel, nComps, centering, jW, bs);
    mgSmoothers_[i]->setParam(2, 2, i ? 0 : 20);
  }

  gstFiller_.resize(amrHier_.size());
  fcFiller_.resize(amrHier_.size());
  for (unsigned i = 0; i < amrHier_.size(); ++i) {
    gstFiller_[i] = amrHier_.createGhostFiller(i);
    fcFiller_[i] = amrHier_.createFuncFiller(i);
  }

  for (unsigned i = 0; i < amrHier_.size(); ++i) {
    levelRes_.push_back(LevelData<Real, Dim>(
        amrHier_.getMesh(i), centering, nComps, amrHier_.getnGhost()));
    levelCorr_.push_back(LevelData<Real, Dim>(
        amrHier_.getMesh(i), centering, nComps, amrHier_.getnGhost()));
    levelAux_.push_back(LevelData<Real, Dim>(
        amrHier_.getMesh(i), centering, nComps, amrHier_.getnGhost()));
  }
  UnitTimer::getInstance().end("initAMRMultigrid");
}

template <int Dim>
void AMRMultigrid<Dim>::solve(
    Vector<LD> &phi,
    const Vector<LD> &rhs,
    const Vector<Vector<Array<Tensor<Real, Dim - 1>, 2 * Dim>>> &cDatas,
    const Vector<Array<char, 2 * Dim>> &bcType,
    Wrapper_Silo<Dim> *dbgSiloOut) {
#ifndef NDEBUG
  // Check the data validation
  assert(phi.size() == amrHier_.size());
  assert(rhs.size() == amrHier_.size());
  for (unsigned i = 0; i < amrHier_.size(); ++i) {
    assert(phi[i].getMesh() == amrHier_.getMesh(i));
    assert(phi[i].getnGhost() == amrHier_.getnGhost());
    assert(phi[i].getnComps() == nComps_);
    assert(rhs[i].getMesh() == amrHier_.getMesh(i));
    assert(rhs[i].getnGhost() == amrHier_.getnGhost());
    assert(rhs[i].getnComps() == nComps_);
    for (unsigned comp = 0; comp < nComps_; ++comp) {
      assert(phi[i].getCentering(comp) == centering_[comp]);
      assert(rhs[i].getCentering(comp) == centering_[comp]);
    }
  }
#endif

  bcType_ = bcType;
  for (auto &mgOp : mgSmoothers_)
    mgOp->setBCType(bcType_);

  constexpr int q_norm = 0;
  constexpr int numRatioRecord = 3;
  Vector<Real> initRes, curRes[2];
  Real ratioRec[numRatioRecord];

  for (int iter = 0; iter < maxIter_; ++iter) {
    for (unsigned i = 0; i < amrHier_.size(); ++i) {
      mgSmoothers_[i]->computeResidual(levelRes_[i], phi[i], rhs[i]);
      levelCorr_[i].memset(0);
      levelAux_[i].memset(0);
    }
    amrIntergridOp_->averageToCoarse(levelRes_);

    if (dbgSiloOut != nullptr) {
      char varname[25];
      sprintf(varname, "residualIter%03d", iter);
      dbgSiloOut->putAMRScalar(levelRes_, varname);
      sprintf(varname, "solutionIter%03d", iter);
      dbgSiloOut->putAMRScalar(phi, varname);
      sprintf(varname, "LphiIter%03d", iter);
      auto Lphi = phi;
      for (unsigned i = 0; i < amrHier_.size(); ++i)
        mgSmoothers_[i]->apply(Lphi[i], phi[i]);
      amrIntergridOp_->averageToCoarse(Lphi);
      dbgSiloOut->putAMRScalar(Lphi, varname);
    }

    if (!iter) {
      initRes = amrIntergridOp_->computeNorm(levelRes_, q_norm);
      curRes[0] = curRes[1] = initRes;
    } else {
      Real maxRelRes = 0, maxRatio = 0.;
      curRes[0] = curRes[1];
      curRes[1] = amrIntergridOp_->computeNorm(levelRes_, q_norm);
      for (unsigned comp = 0; comp < nComps_; ++comp) {
        maxRelRes = std::max(maxRelRes, curRes[1][comp] / initRes[comp]);
        maxRatio = std::max(maxRatio, curRes[0][comp] / curRes[1][comp]);
      }
      ratioRec[(iter - 1) % numRatioRecord] = maxRatio;
      mpicout << "Multigrid iter " << std::setw(2) << iter
              << ", rel. rsd. = " << std::scientific << std::setprecision(3)
              << maxRelRes << ", ratio = " << std::defaultfloat
              << std::setprecision(3) << maxRatio << "\n";
      if (maxRelRes < relTol_)
        break;

      if (iter >= numRatioRecord * 2) {
        Real aveMaxRatio = 0.;
        for (int i = 0; i < numRatioRecord; ++i)
          aveMaxRatio += ratioRec[i];
        if (aveMaxRatio / numRatioRecord < stallTol_)
          break;
      }
    }

    if (useFMG_)
      FMGCycle(phi, rhs, cDatas);
    else {
      VCycle(amrHier_.size() - 1);
      for (unsigned i = 0; i < amrHier_.size(); ++i)
        phi[i] = phi[i] + levelCorr_[i];
      amrIntergridOp_->fillGhosts(phi, cDatas, bcType_);
    }
  }
}

template <int Dim>
void AMRMultigrid<Dim>::fillGhostOnly(int level, LD &fData) {
  for (unsigned comp = 0; comp < nComps_; ++comp)
    gstFiller_[level].fillGhosts(fData, bcType_, comp);
  fData.exchangeAll();
}

template <int Dim>
void AMRMultigrid<Dim>::fillGhost(int level,
                                  LD &fData,
                                  const LD &cData,
                                  LD &cAux) {
  fillGhostOnly(level, fData);
  cAux = cData;
  amrIntergridOp_->averageToCoarse(fData, cAux, level);
  cAux.exchangeAll();
  amrIntergridOp_->interpolateToFineGhost(cAux, fData, level);
}

template <int Dim>
void AMRMultigrid<Dim>::VCycle(int level) {
  if (level == 0) {
    fillGhostOnly(level, levelCorr_[level]);
    mgSmoothers_[level]->relax(
        levelCorr_[level], levelRes_[level], levelAux_[level]);
    levelCorr_[level] = levelAux_[level];
    fillGhostOnly(level, levelCorr_[level]);
    return;
  }

  fillGhost(
      level, levelCorr_[level], levelCorr_[level - 1], levelAux_[level - 1]);
  mgSmoothers_[level]->relax(
      levelCorr_[level], levelRes_[level], levelAux_[level]);

  fillGhost(
      level, levelAux_[level], levelCorr_[level - 1], levelAux_[level - 1]);
  mgSmoothers_[level]->computeResidual(
      levelCorr_[level], levelAux_[level], levelRes_[level]);

  amrIntergridOp_->averageToCoarse(
      levelCorr_[level], levelRes_[level - 1], level);
  VCycle(level - 1);
  amrIntergridOp_->constantInterpolateIncr(
      levelCorr_[level - 1], levelAux_[level], level);

  fillGhost(
      level, levelAux_[level], levelCorr_[level - 1], levelAux_[level - 1]);
  mgSmoothers_[level]->relax(
      levelAux_[level], levelRes_[level], levelCorr_[level]);

  fillGhost(
      level, levelCorr_[level], levelCorr_[level - 1], levelAux_[level - 1]);
}

template <int Dim>
void AMRMultigrid<Dim>::FMGCycle(
    Vector<LD> &phi,
    const Vector<LD> &rhs,
    const Vector<Vector<Array<Tensor<Real, Dim - 1>, 2 * Dim>>> &cDatas) {
  mgSmoothers_[0]->switchFMG(true);
  VCycle(0);
  mgSmoothers_[0]->switchFMG(false);

  for (unsigned fmgIter = 0; fmgIter < amrHier_.size(); ++fmgIter) {
    if (fmgIter) {
      for (unsigned i = 0; i < amrHier_.size(); ++i) {
        mgSmoothers_[i]->computeResidual(levelRes_[i], phi[i], rhs[i]);
        levelCorr_[i].memset(0);
        levelAux_[i].memset(0);
      }
      amrIntergridOp_->averageToCoarse(levelRes_);
      VCycle(fmgIter);
    }

    for (unsigned i = fmgIter; i + 1 < amrHier_.size(); ++i) {
      levelCorr_[i + 1].memset(0);
      amrIntergridOp_->autoInterpolateIncr(
          levelCorr_[i], levelCorr_[i + 1], i + 1, FMGInterpOd_);
      for (unsigned comp = 0; comp < nComps_; ++comp)
        gstFiller_[i + 1].fillGhosts(levelCorr_[i + 1], bcType_, comp);
      amrIntergridOp_->interpolateToFineGhost(
          levelCorr_[i], levelCorr_[i + 1], i + 1);
      levelCorr_[i + 1].exchangeAll();
    }

    for (unsigned i = 0; i < amrHier_.size(); ++i)
      phi[i] = phi[i] + levelCorr_[i];
    amrIntergridOp_->fillGhosts(phi, cDatas, bcType_);
  }
}

template class AMRMultigrid<2>;