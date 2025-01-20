#pragma once

#include "AMRTools/LevelData.h"

#include <AMRTools/ProblemDomain.h>
#include <FiniteDiff/LevelOp.h>

template <typename T>
class LinearSolver {
  // public:
  //   /**
  //    * @brief Compute dst = rhs - L(src)
  //    */
  //   virtual void computeResidual(T &dst, const T &src, const T &rhs) const;

  //   /**
  //    * @brief Compute ||aData||_q
  //    */
  //   virtual void computeNorm(const T &aData, int q) const;

  //   /**
  //    * @brief Relax L(phi)=rhs for one times.
  //    */
  //   virtual void relax(const T &phi, const T &rhs, T &smoothed) const = 0;

  //   /**
  //    * @brief Solve L(phi)=rhs.
  //    */
  //   virtual void solve(T &phi, const T &rhs) const = 0;
};

template <int Dim>
class LinearSolver<LevelData<Real, Dim>> {
public:
  template <class U>
  using Vector = std::vector<U>;

protected:
  ProblemDomain<Dim> domain_;
  LevelOp<Dim> lvOp_;

public:
  LinearSolver(const ProblemDomain<Dim> &domain,
               Communicator comm = MPI_COMM_WORLD) :
      domain_(domain), lvOp_(domain, comm){};

  /**
   * @brief compute dst = L(src)
   */
  virtual void apply(LevelData<Real, Dim> &dst,
                     const LevelData<Real, Dim> &src) const {
    throw std::runtime_error("LinearSolver:: no apply() can be called");
  }

  /**
   * @brief Compute dst = rhs - L(src)
   */
  virtual void computeResidual(LevelData<Real, Dim> &dst,
                               const LevelData<Real, Dim> &src,
                               const LevelData<Real, Dim> &rhs) const;

  /**
   * @brief Compute ||aData||_q
   */
  virtual Vector<Real> computeNorm(const LevelData<Real, Dim> &aData,
                                   int q) const {
    return lvOp_.computeNorm(aData, q);
  }

  /**
   * @brief Relax L(phi)=rhs for one times.
   *
   * @param phi: initial guess
   * @param rhs: right hand side
   * @param smoothed: smoothed result
   */
  virtual void relax(const LevelData<Real, Dim> &phi,
                     const LevelData<Real, Dim> &rhs,
                     LevelData<Real, Dim> &smoothed) {
    throw std::runtime_error("LinearSolver:: no relax() can be called");
  };

  /**
   * @brief Solve L(phi)=rhs.
   */
  virtual void solve(LevelData<Real, Dim> &phi,
                     const LevelData<Real, Dim> &rhs) {
    throw std::runtime_error("LinearSolver:: no solve() can be called");
  };
};