#include "AMRTools/ProblemDomain.h"

template <int Dim>
ProblemDomain<Dim> ProblemDomain<Dim>::getRefined(int refRatio) const {
  ProblemDomain<Dim> refPD;
  refPD.layout_ = layout_.getRefined(refRatio);
  refPD.dx_ = dx_ / (Real)refRatio;
  refPD.nGhost_ = nGhost_;
  refPD.isInner_ = isInner_;
  return refPD;
}

template <int Dim>
ProblemDomain<Dim> ProblemDomain<Dim>::getCoarsened(int refRatio) const {
  ProblemDomain<Dim> csnPD;
  csnPD.layout_ = layout_.getCoarsened(refRatio);
  csnPD.dx_ = dx_ * (Real)refRatio;
  csnPD.nGhost_ = nGhost_;
  csnPD.isInner_ = isInner_;
  return csnPD;
}

template <int Dim>
void ProblemDomain<Dim>::initIsInner() {
  isInner_.resize(size());
  for (auto it = layout_.begin(); it.ok(); ++it) {
    isInner_[it.index()][0] =
        layout_.whichBox((iVec){it->lo()[0] - 1, it->lo()[1]}).has_value();
    isInner_[it.index()][1] =
        layout_.whichBox((iVec){it->hi()[0] + 1, it->lo()[1]}).has_value();
    isInner_[it.index()][2] =
        layout_.whichBox((iVec){it->lo()[0], it->lo()[1] - 1}).has_value();
    isInner_[it.index()][3] =
        layout_.whichBox((iVec){it->lo()[0], it->hi()[1] + 1}).has_value();
  }
}

template class ProblemDomain<2>;