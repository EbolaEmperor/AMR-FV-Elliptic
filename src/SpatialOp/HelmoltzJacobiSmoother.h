#pragma once

#include "AMRTools/LevelData.h"
#include "AMRTools/ProblemDomain.h"
#include "FiniteDiff/LevelOp.h"
#include "SpatialOp/LinearSolver.h"

template <int Dim, int Order = 4>
class HelmoltzJacobiSmoother : public LinearSolver<LevelData<Real, Dim>> {
  // {
public:
  using LD = LevelData<Real, Dim>;
  using BaseClass = LinearSolver<LD>;

protected:
  // Helmoltz coefficient: H(u) = alpha * u + beta * Lap(u)
  const Real alpha_;
  const Real beta_;
  const Real jacobiWeight_;

  LevelOp<Dim> lvOp_;

public:
  HelmoltzJacobiSmoother(const ProblemDomain<Dim> &pd,
                         Real alpha,
                         Real beta,
                         Real jW) :
      BaseClass(pd),
      alpha_(alpha),
      beta_(beta),
      jacobiWeight_(jW),
      lvOp_(pd) {}

public:
  /**
   * @brief compute dst = H(src)
   */
  void apply(LD &dst, const LD &src) const;

  /**
   * @brief smooth equation H(*)=rhs for one times
   *
   * @param phi: initial guess
   * @param rhs: right hand side
   * @param smoothed: the smoothed result
   */
  void relax(const LD &phi, const LD &rhs, LD &smoothed);

  /**
   * @brief smooth equation H(*)=rhs for one times
   *
   * @param phi: initial guess and the smoothed result
   * @param rhs: right hand side
   */
  void solve(LD &phi, const LD &rhs) {
    LD midres = phi;
    relax(midres, rhs, phi);
  }
};