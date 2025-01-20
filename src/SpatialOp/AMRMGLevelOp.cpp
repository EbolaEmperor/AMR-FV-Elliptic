#include "SpatialOp/AMRMGLevelOp.h"

#ifndef NDEBUG
#include "AMRTools/Wrapper_Silo.h"
#endif

#include "AMRTools/LevelDataExpr.h"
#include "SpatialOp/HelmoltzJacobiSmoother.h"

template <int Dim>
void AMRMGLevelOp<Dim>::setParam(int preSmooth,
                                 int postSmooth,
                                 int bottomSmooth,
                                 bool useHuangVC) {
  if (!defined_) {
    throw std::runtime_error(
        "AMRMGLevelOp:: Please call setParam() afrer defined.");
  }
  preSmooth_ = preSmooth;
  postSmooth_ = postSmooth;
  bottomSmooth_ = bottomSmooth;
  useHuangVC_ = useHuangVC;
}

template <int Dim>
void AMRMGLevelOp<Dim>::switchFMG(bool useFMG) {
  useFMG_ = useFMG;
}

template <int Dim>
void AMRMGLevelOp<Dim>::define(const DisjointBoxLayout<Dim> &finestMesh,
                               Real alpha,
                               Real beta,
                               unsigned nLevel,
                               int centering,
                               Real jW,
                               std::shared_ptr<LS> bottomSolver) {
  Vector<int> cents(1, centering);
  define(finestMesh, alpha, beta, nLevel, 1, cents, jW, bottomSolver);
}

template <int Dim>
void AMRMGLevelOp<Dim>::define(const DisjointBoxLayout<Dim> &finestMesh,
                               Real alpha,
                               Real beta,
                               unsigned nLevel,
                               unsigned nComps,
                               const Vector<int> &centering,
                               Real jacobiW,
                               std::shared_ptr<LS> bottomSolver) {
  // Only support Cell-Center now...
  for (unsigned i = 0; i < nComps; ++i)
    assert(centering[i] == CellCenter);
  nComps_ = nComps;

  if (defined_) {
    throw std::runtime_error("AMRMGLevelOp:: redefined.");
  }
  defined_ = true;
  nLevel_ = nLevel;

  meshes_.resize(nLevel_);
  meshes_.back() = finestMesh;
  for (int i = meshes_.size() - 1; i > 0; --i) {
    meshes_[i - 1] = meshes_[i].getCoarsened(2);
  }

  smoothers_.resize(nLevel_);
  smoothersOd2_.resize(nLevel_);
  gstFillers_.resize(nLevel_);
  fcFillers_.resize(nLevel_);
  imDomains_.resize(nLevel_);
  auto curPd = BaseClass::domain_;
  for (int i = nLevel_ - 1; i >= 0; --i) {
    imDomains_[i] = curPd;
    smoothers_[i] = std::make_unique<HelmoltzJacobiSmoother<Dim>>(
        HelmoltzJacobiSmoother<Dim>(curPd, alpha, beta, jacobiW));
    smoothersOd2_[i] = std::make_unique<HelmoltzJacobiSmoother<Dim, 2>>(
        HelmoltzJacobiSmoother<Dim, 2>(curPd, alpha, beta, 0.9));
    gstFillers_[i] = GhostFiller<Dim>(
        curPd, meshes_[i], LevelGhostBoxes<Dim>(curPd, meshes_[i]));
    fcFillers_[i] = FuncFiller<Dim>(curPd, meshes_[i]);
    if (i)
      curPd = curPd.getCoarsened(2);
  }

  mgIntergridOps_.resize(nLevel_ - 1);
  for (unsigned i = 0; i < nLevel_ - 1; ++i) {
    mgIntergridOps_[i] =
        std::make_unique<MGIntergrid<Dim>>(MGIntergrid<Dim>());
  }

  bottomSolver_ = bottomSolver;

  for (unsigned i = 0; i < nLevel_; ++i) {
    levelRes_.emplace_back(meshes_[i], centering, nComps);
    levelCorr_.emplace_back(meshes_[i], centering, nComps);
    levelAux_.emplace_back(meshes_[i], centering, nComps);
  }
}

template <int Dim>
void AMRMGLevelOp<Dim>::apply(LD &dst, const LD &src) const {
  if (!defined_) {
    throw std::runtime_error(
        "AMRMGLevelOp:: please call apply() after defined.");
  }
  smoothers_.back()->apply(dst, src);
}

template <int Dim>
void AMRMGLevelOp<Dim>::relax(const LD &phi, const LD &rhs, LD &smoothed) {
  if (!defined_) {
    throw std::runtime_error("AMRMGLevelOp:: used relax() before defined.");
  }
  smoothers_[nLevel_ - 1]->computeResidual(levelRes_.back(), phi, rhs);
  if (useFMG_)
    FMGCycle(nLevel_ - 1);
  else {
    levelCorr_.back().memset(0);
    VCycle(nLevel_ - 1);
  }
  smoothed = phi + levelCorr_.back();
}

template <int Dim>
void AMRMGLevelOp<Dim>::VCycle(int level) {
  auto doSmooth = [&](LD &init, LD &corr) {
    for (int comp = 0; comp < nComps_; ++comp) {
      gstFillers_[level].fillGhostsCFHomo(init, comp, true);
      gstFillers_[level].fillGhosts(init, bcType_, comp);
    }
    init.exchangeAll();
    gstFillers_[level].fillCorners(init);
    for (int comp = 0; comp < nComps_; ++comp) {
      smoothers_[level]->relax(init, levelRes_[level], corr);
    }
    corr.exchangeAll();
  };

  for (int i = 0; i < preSmooth_; i += 2) {
    doSmooth(levelCorr_[level], levelAux_[level]);
    doSmooth(levelAux_[level], levelCorr_[level]);
  }

  if (level == 0) {
    for (int i = 0; i < bottomSmooth_; i += 2) {
      doSmooth(levelCorr_[level], levelAux_[level]);
      doSmooth(levelAux_[level], levelCorr_[level]);
    }
    if (bottomSolver_ != nullptr)
      bottomSolver_->solve(levelCorr_[level], levelRes_[level]);
  } else {
    for (int comp = 0; comp < nComps_; ++comp) {
      gstFillers_[level].fillGhostsCFHomo(levelCorr_[level], comp, true);
      gstFillers_[level].fillGhosts(levelCorr_[level], bcType_, comp);
    }
    levelCorr_[level].exchangeAll();

    smoothers_[level]->computeResidual(
        levelAux_[level], levelCorr_[level], levelRes_[level]);

    if (useHuangVC_) {
      auto tmpCorr = levelCorr_[level];
      auto tmpRes = levelRes_[level];
      levelRes_[level] = levelAux_[level];
      levelCorr_[level].memset(0);
      VCycleOd2(level);
      levelCorr_[level] = levelCorr_[level] + tmpCorr;
      levelRes_[level] = tmpRes;
    } else {
      mgIntergridOps_[level - 1]->applyRestrict(levelAux_[level],
                                                levelRes_[level - 1]);

      levelCorr_[level - 1].memset(0);
      VCycle(level - 1);

      mgIntergridOps_[level - 1]->applyInterpolation(levelCorr_[level - 1],
                                                     levelAux_[level]);
      levelCorr_[level] = levelCorr_[level] + levelAux_[level];
    }
  }

  for (int i = 0; i < postSmooth_; i += 2) {
    doSmooth(levelCorr_[level], levelAux_[level]);
    doSmooth(levelAux_[level], levelCorr_[level]);
  }
}

template <int Dim>
void AMRMGLevelOp<Dim>::FMGCycle(int level) {
  levelCorr_[level].memset(0);
  if (level == 0)
    return VCycle(0);
  mgIntergridOps_[level - 1]->applyRestrict(levelRes_[level],
                                            levelRes_[level - 1]);
  FMGCycle(level - 1);
  mgIntergridOps_[level - 1]->applyInterpolation(levelCorr_[level - 1],
                                                 levelCorr_[level]);
  VCycle(level);
}

template <int Dim>
void AMRMGLevelOp<Dim>::VCycleOd2(int level) {
  auto doSmooth = [&](LD &init, LD &corr) {
    for (int comp = 0; comp < nComps_; ++comp) {
      gstFillers_[level].fillGhostsCFHomo(init, comp, true);
      gstFillers_[level].fillGhostsOd2(init, bcType_, comp);
    }
    init.exchangeAll();
    for (int comp = 0; comp < nComps_; ++comp) {
      smoothersOd2_[level]->relax(init, levelRes_[level], corr);
    }
    corr.exchangeAll();
  };

  for (int i = 0; i < preSmooth_ / 2; i += 2) {
    doSmooth(levelCorr_[level], levelAux_[level]);
    doSmooth(levelAux_[level], levelCorr_[level]);
  }

  if (level == 0) {
    for (int i = 0; i < bottomSmooth_; i += 2) {
      doSmooth(levelCorr_[level], levelAux_[level]);
      doSmooth(levelAux_[level], levelCorr_[level]);
    }
    if (bottomSolver_ != nullptr)
      bottomSolver_->solve(levelCorr_[level], levelRes_[level]);
  } else {
    for (int comp = 0; comp < nComps_; ++comp) {
      gstFillers_[level].fillGhostsCFHomo(levelCorr_[level], comp, true);
      gstFillers_[level].fillGhostsOd2(levelCorr_[level], bcType_, comp);
    }
    levelCorr_[level].exchangeAll();

    smoothersOd2_[level]->computeResidual(
        levelAux_[level], levelCorr_[level], levelRes_[level]);

    mgIntergridOps_[level - 1]->applyRestrict(levelAux_[level],
                                              levelRes_[level - 1]);

    levelCorr_[level - 1].memset(0);
    VCycleOd2(level - 1);

    mgIntergridOps_[level - 1]->applyInterpolation(levelCorr_[level - 1],
                                                   levelAux_[level]);
    levelCorr_[level] = levelCorr_[level] + levelAux_[level];
  }

  for (int i = 0; i < postSmooth_ / 2; i += 2) {
    doSmooth(levelCorr_[level], levelAux_[level]);
    doSmooth(levelAux_[level], levelCorr_[level]);
  }
}

template class AMRMGLevelOp<2>;