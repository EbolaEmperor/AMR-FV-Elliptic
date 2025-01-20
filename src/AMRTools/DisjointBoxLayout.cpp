/**
 * @file DisjointBoxLayout.cpp
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 * @copyright Copyright (c) 2024 Wenchong Huang
 *
 */

#include "AMRTools/DisjointBoxLayout.h"

template <int Dim>
std::optional<int> DisjointBoxLayout<Dim>::whichBox(const iVec &idx) const {
  for (auto it = begin(); it.ok(); ++it) {
    if (it->contain(idx)) {
      return it.index();
    }
  }
  return std::nullopt;
};

template <int Dim>
void DisjointBoxLayout<Dim>::checkDisjoint() const {
  for (unsigned i = 0; i < boxes_.size(); i++) {
    for (unsigned j = i + 1; j < boxes_.size(); j++) {
      if (!(boxes_[i] & boxes_[j]).empty()) {
        throw std::runtime_error(
            "Your boxes are NOT DISJOINT in a DisjointLoxLayout!");
      }
    }
  }
};

template <int Dim>
void DisjointBoxLayout<Dim>::sort() {
  checkDisjoint();

  auto cmpiVec = [](const iVec &i, const iVec &j) {
    return i[0] < j[0] || (i[0] == j[0] && i[1] < j[1]);
  };

  auto cmp = [&](const int &i, const int &j) {
    return cmpiVec(getBox(i).lo(), getBox(j).lo());
  };

  Vector<int> indices(size());
  for (unsigned i = 0; i < size(); i++)
    indices[i] = i;
  std::sort(indices.begin(), indices.end(), cmp);

  Vector<Box<Dim>> sortedBoxes(size());
  for (unsigned i = 0; i < size(); i++) {
    sortedBoxes[i] = getBox(indices[i]);
  }

  boxes_ = std::move(sortedBoxes);

  if (!procs_.empty()) {
    Vector<int> sortedProcs(size());
    for (unsigned i = 0; i < size(); i++) {
      sortedProcs[i] = getProcID(indices[i]);
    }
    procs_ = std::move(sortedProcs);
  }
};

template <int Dim>
DisjointBoxLayout<Dim> DisjointBoxLayout<Dim>::getRefined(int refRatio) const {
  Vector<Box<Dim>> boxes(size());
  Vector<int> procs = procs_;

  for (auto it = begin(); it.ok(); ++it) {
    boxes[it.index()] = it->getRefined(refRatio);
  }

  // To avoid a meaningless sort,
  // we donot use the constructor of DisjointBoxLayout
  DisjointBoxLayout result;
  result.boxes_ = std::move(boxes);
  result.procs_ = std::move(procs);
  return result;
}

template <int Dim>
DisjointBoxLayout<Dim> DisjointBoxLayout<Dim>::getCoarsened(
    int refRatio) const {
  Vector<Box<Dim>> boxes(size());
  Vector<int> procs = procs_;

  for (auto it = begin(); it.ok(); ++it) {
    // Check whether the domain could be coarsened by the given ratio.
    for (int d = 0; d < Dim; d++)
      if (it->lo()[d] % refRatio || (it->hi()[d] + 1) % refRatio) {
        std::stringstream errmsg;
        errmsg << "Cannot coarsen box " << (*it) << " by ratio " << refRatio;
        throw std::runtime_error(errmsg.str());
      }

    boxes[it.index()] = it->getCoarsened(refRatio);
  }

  // To avoid a meaningless sort,
  // we donot use the constructor of DisjointBoxLayout
  DisjointBoxLayout result;
  result.boxes_ = std::move(boxes);
  result.procs_ = std::move(procs);
  return result;
}

template <int Dim>
std::optional<int> DisjointBoxLayout<Dim>::getDataID(int box_id) const {
  int data_id = -1;
  for (int i = 0; i < box_id; ++i)
    if (procs_[i] == procs_[box_id])
      ++data_id;
  if (data_id >= 0) {
    return std::optional<int>(data_id);
  }
  return std::nullopt;
}

template class DisjointBoxLayout<2>;