#pragma once

#include "AMRTools/AMRMeshHierachy.h"
#include "AMRTools/LevelData.h"
#include "AMRTools/Wrapper_Silo.h"
#include "FiniteDiff/AMRIntergridOp.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "SpatialOp/AMRMGLevelOp.h"
#include "SpatialOp/LinearSolver.h"

#include <array>
#include <memory>
#include <vector>

template <int Dim>
class AMRMultigrid {
public:
  using LD = LevelData<Real, Dim>;
  using LS = LinearSolver<LD>;
  template <typename T>
  using Vector = std::vector<T>;
  template <typename T, int R>
  using Array = std::array<T, R>;

protected:
  bool defined_;
  const AMRMeshHierachy<Dim> &amrHier_;
  std::unique_ptr<AMRIntergridOp<Dim>> amrIntergridOp_;

  Vector<std::unique_ptr<AMRMGLevelOp<Dim>>> mgSmoothers_;
  Vector<GhostFiller<Dim>> gstFiller_;
  Vector<FuncFiller<Dim>> fcFiller_;

  Vector<LD> levelRes_;
  Vector<LD> levelCorr_;
  Vector<LD> levelAux_;

  int maxIter_;
  Real relTol_;
  Real stallTol_;

  bool useFMG_;
  int FMGInterpOd_;

  unsigned nComps_;
  Vector<int> centering_;

  Vector<Array<char, 2 * Dim>> bcType_;

public:
  AMRMultigrid(const AMRMeshHierachy<Dim> &amrHier) :
      defined_(false),
      amrHier_(amrHier),
      maxIter_(40),
      relTol_(1e-14),
      stallTol_(1e-14),
      useFMG_(false) {}

public:
  /**
   * @breif: Define a AMRMultigrid for a certain type of LevelData
   *
   * @param alpha, beta: Helmoltz coefficients
   * @param nComps: the number of components
   * @param centering: the centering of each component
   * @param bottomNumLevel: the number of implicit MG levels
   *                        under the bottom AMR level.
   * @param jacobiW: the weight of jacobi relaxation
   * @param bottomSolver: the bottom solver for the bottom implicit MG level.
   *
   * @note note that bottomSolver is the solver
   *       for the most bottom implicit MG level
   *       but not the bottom AMR level !!!
   */
  void define(Real alpha,
              Real beta,
              unsigned nComps,
              const Vector<int> &centering,
              int bottomNumLevel,
              Real jacobiW = 0.5,
              std::shared_ptr<LS> bottomSolver = nullptr);

  /**
   * @breif: Define a AMRMultigrid for a LevelData with only 1 component.
   *
   * @param alpha, beta: Helmoltz coefficients
   * @param centering: the centering of the only component
   * @param bottomNumLevel: the number of implicit MG levels
   *                        under the bottom AMR level.
   * @param bottomSolver: the bottom solver for the bottom implicit MG level.
   */
  void define(Real alpha,
              Real beta,
              int centering,
              int bottomNumLevel,
              Real jacobiW = 0.5,
              std::shared_ptr<LS> bottomSolver = nullptr);

  /**
   * @breif: Set parameters of AMR multigrid algorithm
   */
  void setParam(int preSmooth,
                int postSmooth,
                int bottomSmooth,
                int maxIter,
                Real relTol,
                Real stallTol,
                bool useFMG = false,
                int FMGInterpOd = 1);

  /**
   * @breif: Solve the eqaution on the AMR mesh
   *
   * @param phi: the numerical solution.
   * @param rhs: the rignt hand side.
   * @param cDatas[l][d][f]: the boundary conditions on the f-th face
   *                         of the d-th domain on the l-th level.
   * @param bcType[d][f]: the boundary condition type of the f-th face
   *                      of the d-th domain.
   * @param dbgSiloOut: if you want to see the middle result,
   *                    put a Wrapper_Silo<Dim> pointer here.
   */
  void solve(
      Vector<LD> &phi,
      const Vector<LD> &rhs,
      const Vector<Vector<Array<Tensor<Real, Dim - 1>, 2 * Dim>>> &cDatas,
      const Vector<Array<char, 2 * Dim>> &bcType,
      Wrapper_Silo<Dim> *dbgSiloOut = nullptr);

protected:
  /**
   * @breif: The AMR V-cycle algorithm.
   */
  void VCycle(int level);

  /**
   * @breif: The AMR FMG-cycle algorithm.
   */
  void FMGCycle(
      Vector<LD> &phi,
      const Vector<LD> &rhs,
      const Vector<Vector<Array<Tensor<Real, Dim - 1>, 2 * Dim>>> &cDatas);

  /**
   * @breif: fill ghost and coincide with the coarse data
   */
  void fillGhost(int level, LD &fData, const LD &cData, LD &cAux);

  /**
   * @breif: fill ghost
   */
  void fillGhostOnly(int level, LD &fData);
};