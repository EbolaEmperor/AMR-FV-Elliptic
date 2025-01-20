#pragma once

#include "AMRTools/LevelData.h"
#include "AMRTools/ProblemDomain.h"
#include "Core/MPI.h"

#include <vector>

template <int Dim>
class LevelOp {
public:
  using iVec = Vec<int, Dim>;
  using LD = LevelData<Real, Dim>;

  LevelOp() {}

  LevelOp(const ProblemDomain<Dim> &domain,
          Communicator comm = MPI_COMM_WORLD) :
      domain_(domain), comm_(comm) {}

public:
  /**
   * @brief Compute ||aData||_q
   * @return the q-norm of each comonent of aData.
   */
  std::vector<Real> computeNorm(const LevelData<Real, Dim> &aData,
                                int q) const;

  /**
   * @brief Apply the weighted Jacobi iteration to the equations (alpha * I +
   * beta * Lap) phi = rhs,
   * @param w The weight of the Jacobi iteration.
   */
  void relaxJacobi(const LD &phi,
                   const LD &rhs,
                   LD &smoothed,
                   Real alpha = 0.,
                   Real beta = 1.,
                   Real w = .5) const;

  void relaxJacobiOd2(const LD &phi,
                      const LD &rhs,
                      LD &smoothed,
                      Real alpha = 0.,
                      Real beta = 1.,
                      Real w = .5) const;

  void relaxSOROd2(const LD &phi,
                   const LD &rhs,
                   LD &smoothed,
                   Real alpha = 0.,
                   Real beta = 1.,
                   Real w = .5) const;

  /**
   * @brief compute lap = Lap(phi)
   */
  void computeLaplacian(const LD &phi, LD &lap) const;

  void computeLaplacianOd2(const LD &phi, LD &lap) const;

  /**
   * @brief compute helm = alpha * phi + beta * Lap(phi)
   */
  void computeHelmoltz(const LD &phi, LD &helm, Real alpha, Real beta) const;

  void computeHelmoltzOd2(const LD &phi,
                          LD &helm,
                          Real alpha,
                          Real beta) const;

  /**
   * @brief compute grad = Grad(phi)
   */
  void computeGradient(const LD &phi, LD &grad) const;

  /**
   * @brief compute d2phi[component d] = d^2 phi / d x_d^2
   */
  void computeD2(const LD &phi, LD &d2phi) const;

  /**
   * @brief compute div = Div(u)
   */
  void computeDivergence(const LD &u, LD &div) const;

  /**
   * @brief compute curl = Curl(u)
   */
  void computeCurl(const LD &u, LD &curl) const;

  /**
   * @brief compute magu = |u|
   * @param u: a cell-centered or node-centered vector.
   */
  void computeMagnitude(const LD &u, LD &magu) const;

  /**
   * @brief filter u from face-centered to cell-centered
   */
  void filterFace2Cell(const LD &u, LD &cellu) const;

  /**
   * @brief filter u from face-centered to cell-centered with a stencil
   *        of order 2, so that the ghosts of u are not necessary.
   */
  void filterFace2CellOd2(const LD &u, LD &cellu) const;

  /**
   * @brief filter u from cell-centered to face-centered.
   *        4th order. Vector-value only.
   */
  void filterCell2Face(const LD &u, LD &faceu) const;

  /**
   * @brief filter u from cell-centered to face-centered.
   *        2nd order. Vector-value only.
   */
  void filterCell2FaceOd2(const LD &u, LD &cellu) const;

  /**
   * @brief compute refine tags by err.
   *
   * @param err: a variable whose absolute value may indicate
   *             the distrubution of errors.
   *             eg. magnitude, gradient, or divergence.
   * @param thereshold: ranged in [0,1]. If abs(err) in some control element
   *       (maybe a cell, a face or a node, depends on the centering of err)
   *       is greater than thereshold*max(abs(err)), then tag the related cell.
   */
  void computeTags(const LD &err,
                   LevelData<bool, 2> &tags,
                   Real thereshold) const;

protected:
  void computeGradientCell2Face(const LD &phi, LD &grad) const;

  void computeGradientCell2Cell(const LD &phi, LD &grad) const;

protected:
  std::vector<Real> handleOverlappingInNorm(const LevelData<Real, Dim> &aData,
                                            int q) const;

protected:
  ProblemDomain<Dim> domain_;
  Communicator comm_;
};