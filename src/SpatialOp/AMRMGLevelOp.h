#pragma once

#include "AMRTools/DisjointBoxLayout.h"
#include "AMRTools/LevelData.h"
#include "AMRTools/ProblemDomain.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "FiniteDiff/MGIntergrid.h"
#include "SpatialOp/LinearSolver.h"

#include <iostream>
#include <memory>
#include <vector>

template <int Dim>
class AMRMGLevelOp : public LinearSolver<LevelData<Real, Dim>> {
public:
  using LD = LevelData<Real, Dim>;
  using LS = LinearSolver<LD>;
  using BaseClass = LS;
  template <typename T>
  using Vector = std::vector<T>;
  template <typename T, int R>
  using Array = std::array<T, R>;

  using BaseClass::computeResidual;

protected:
  bool defined_;
  unsigned nLevel_;

  Vector<ProblemDomain<Dim>> imDomains_;
  Vector<DisjointBoxLayout<Dim>> meshes_;

  Vector<GhostFiller<Dim>> gstFillers_;
  Vector<FuncFiller<Dim>> fcFillers_;

  Vector<std::unique_ptr<LS>> smoothers_;
  Vector<std::unique_ptr<LS>> smoothersOd2_;
  Vector<std::unique_ptr<MGIntergrid<Dim>>> mgIntergridOps_;
  std::shared_ptr<LS> bottomSolver_;

  Vector<LD> levelRes_;
  Vector<LD> levelCorr_;
  Vector<LD> levelAux_;

  int nComps_;

  int preSmooth_;
  int postSmooth_;
  int bottomSmooth_;

  bool useFMG_;
  bool useHuangVC_;

  Vector<Array<char, 2 * Dim>> bcType_;

public:
  AMRMGLevelOp(const ProblemDomain<Dim> &pd) :
      BaseClass(pd),
      defined_(false),
      nLevel_(0),
      preSmooth_(2),
      postSmooth_(2),
      bottomSmooth_(0),
      useFMG_(false) {}

public:
  /**
   * @breif: Define a AMRMGLevelOp
   *
   * @param finestDomain: the domain of a AMR level
   * @param finestMesh: the mesh of a AMR level
   * @param alpha, beta: Helmoltz coefficients
   * @param nLevel: the number of implicit MG levels
   * @param nComps: the number of components
   * @param centering: the centering of each component
   * @param jacobiW: the weight of jacobi relaxation
   * @param bottomSolver: the bottom solver
   */
  void define(const DisjointBoxLayout<Dim> &finestMesh,
              Real alpha,
              Real beta,
              unsigned nLevel,
              unsigned nComps,
              const Vector<int> &centering,
              Real jacobiW = 0.5,
              std::shared_ptr<LS> bottomSolver = nullptr);

  /**
   * @breif: Define a AMRMGLevelOp for one component LD.
   *
   * @param finestDomain: the domain of a AMR level
   * @param finestMesh: the mesh of a AMR level
   * @param alpha, beta: Helmoltz coefficients
   * @param nLevel: the number of implicit MG levels
   * @param centering: the centering of the LD
   * @param jacobiW: the weight of jacobi relaxation
   * @param bottomSolver: the bottom solver
   */
  void define(const DisjointBoxLayout<Dim> &finestMesh,
              Real alpha,
              Real beta,
              unsigned nLevel,
              int centering = CellCenter,
              Real jacobiW = 0.5,
              std::shared_ptr<LS> bottomSolver = nullptr);

  /**
   * @breif: Set parameters of multigrid algorithm
   */
  void setParam(int preSmooth,
                int postSmooth,
                int bottomSmooth = 0,
                bool useHuangVC = false);

  /**
   * @breif: Switch to the FMG-cycle or switch back to V-cycle.
   */
  void switchFMG(bool useFMG);

  /**
   * @brief Relax H(phi)=rhs with multigrid V-cycle for one times.
   *
   * @param phi: initial guess
   * @param rhs: right hand side
   * @param smoothed: smoothed result
   */
  void relax(const LD &phi, const LD &rhs, LD &smoothed);

  /**
   * @brief compute dst = H(src)
   */
  void apply(LD &dst, const LD &src) const;

  void setBCType(const Vector<Array<char, 2 * Dim>> &bcType) {
    bcType_ = bcType;
  }

protected:
  /**
   * @breif: Multigrid V-cycle algorithm
   */
  void VCycle(int level);

  /**
   * @breif: 2nd-order multigrid V-cycle algorithm
   */
  void VCycleOd2(int level);

  /**
   * @breif: Multigrid FMG-cycle algorithm
   */
  void FMGCycle(int level);
};