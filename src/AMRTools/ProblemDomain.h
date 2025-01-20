/**
 * @file ProblemDomain.h
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 * @copyright Copyright (c) 2024 Wenchong Huang
 *
 */

#pragma once

#include "AMRTools/DisjointBoxLayout.h"

#include <array>

template <int Dim>
class ProblemDomain {
public:
  template <typename U>
  using Vector = std::vector<U>;
  using iVec = Vec<int, Dim>;
  using rVec = Vec<Real, Dim>;

protected:
  // Representing the domain by a list of disjoint boxes.
  DisjointBoxLayout<Dim> layout_;

  // Mesh spacing of the problem domain.
  rVec dx_;

  // Cordinate of the left-lower coner
  rVec x0_;

  // The bandwidth of the ghost layer
  int nGhost_;

  // isInner_[i][j] means whether the j-th face of the i-th domain box
  // is an inner face.
  std::vector<std::array<bool, 4>> isInner_;

public:
  DisjointBoxLayout<Dim>::Iterator begin() const { return layout_.begin(); }

public:
  ProblemDomain() {}

  ProblemDomain(const Vector<Box<Dim>> &boxes,
                rVec dx,
                rVec x0 = 0.,
                int nGhost = 2) :
      layout_(boxes), dx_(std::move(dx)), x0_(std::move(x0)), nGhost_(nGhost) {
    initIsInner();
  }

  // If the input is an expiring value, use std::move to avoid deep copy.
  ProblemDomain(Vector<Box<Dim>> &&boxes,
                rVec dx,
                rVec x0 = 0.,
                int nGhost = 2) :
      layout_(boxes), dx_(std::move(dx)), x0_(std::move(x0)), nGhost_(nGhost) {
    initIsInner();
  }

  const DisjointBoxLayout<Dim> &getLayout() const { return layout_; }

  const rVec &getDx() const { return dx_; }

  const rVec &getX0() const { return x0_; }

  int getNumGhosts() const { return nGhost_; }

  unsigned size() const { return layout_.size(); }

  bool isInnerFace(int i, int j) { return isInner_[i][j]; }

  /**
   * @brief refine all of the boxes by the given ratio
   *
   * @param refRatio : the refine ratio
   * @return the refined ProblemDomain
   */
  ProblemDomain<Dim> getRefined(int refRatio = 2) const;

  /**
   * @brief coarsen all of the boxes by the given ratio
   *
   * @param refRatio : the refine ratio
   * @return the coarsened ProblemDomain
   */
  ProblemDomain<Dim> getCoarsened(int refRatio = 2) const;

protected:
  // Initialize isInner_
  void initIsInner();
};

template <int Dim>
std::ostream &operator<<(std::ostream &os, const ProblemDomain<Dim> &pd) {
  os << "ProblemDomain::" << pd.getLayout();
  os << "x0 = " << pd.getX0() << ", dx = " << pd.getDx() << std::endl;
  return os;
}