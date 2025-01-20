/**
 * @file DisjointBoxLayout.h
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 * @copyright Copyright (c) 2024 Wenchong Huang
 *
 */

#pragma once

#include "AMRTools/BaseIterator.h"
#include "Core/Box.h"
#include "Core/MPI.h"
#include "Core/Vec.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <vector>

template <int Dim>
class DisjointBoxLayout {
public:
  template <typename U>
  using Vector = std::vector<U>;
  using rVec = Vec<Real, Dim>;
  using iVec = Vec<int, Dim>;

protected:
  /**
   * @brief A list of disjoint boxes.
   *
   */
  Vector<Box<Dim>> boxes_;

  Vector<int> procs_;

public:
  class Iterator : public BaseIterator<const Vector<Box<Dim>>> {
  protected:
    using BaseIterator<const Vector<Box<Dim>>>::object_;
    using BaseIterator<const Vector<Box<Dim>>>::current_;

  public:
    using BaseIterator<const Vector<Box<Dim>>>::BaseIterator;
    const Box<Dim> &operator*() { return object_.at(current_); }
    const Box<Dim> *operator->() { return &object_.at(current_); }
  };

  Iterator begin() const { return Iterator(boxes_); }

public:
  DisjointBoxLayout() {}

  /**
   * @brief Construct a new Disjoint Box Layout object
   *
   * @param boxes: given boxes
   */
  DisjointBoxLayout(const Vector<Box<Dim>> &boxes) : boxes_(boxes) { sort(); }

  DisjointBoxLayout(Vector<Box<Dim>> &&boxes) {
    boxes_ = std::move(boxes);
    sort();
  }

  /**
   * @brief Construct a new Disjoint Box Layout object
   *        with process assignment
   *
   * @param boxes: given boxes
   * @param procs: the process assignment
   */
  DisjointBoxLayout(const Vector<Box<Dim>> &boxes, const Vector<int> procs) :
      boxes_(boxes), procs_(procs) {
    assert(boxes_.size() == procs_.size());
    sort();
  }

  DisjointBoxLayout(Vector<Box<Dim>> &&boxes, Vector<int> &&procs) {
    boxes_ = std::move(boxes);
    procs_ = std::move(procs);
    assert(boxes_.size() == procs_.size());
    sort();
  }

  /**
   * @brief get the number of boxes.
   *
   * @return const unsigned
   */
  const unsigned size() const { return boxes_.size(); }
  /**
   * @brief Get the Box object with given index.
   *
   * @param i given index of the box to get.
   * @return const Box<Dim>&
   */
  const Box<Dim> &getBox(int i) const { return boxes_[i]; }

  /**
   * @brief find the box that contains the given cell.
   *
   * @param idx given index of cell
   * @return std::optional<unsigned> the index of the box found.
   */
  std::optional<int> whichBox(const iVec &idx) const;

  /**
   * @brief check if the index idx is in the DBL.
   *
   * @param idx given index of cell
   * @return bool
   */
  bool contain(const iVec &idx) const { return whichBox(idx).has_value(); }

  /**
   * @brief refine all of the boxes by the given ratio
   *
   * @param refRatio : the refine ratio
   * @return the refined DisjointBoxLayout
   */
  DisjointBoxLayout<Dim> getRefined(int refRatio = 2) const;

  /**
   * @brief coarsen all of the boxes by the given ratio
   *
   * @param refRatio : the coarsen ratio
   * @return the coarsened DisjointBoxLayout
   */
  DisjointBoxLayout<Dim> getCoarsened(int refRatio = 2) const;

  /**
   * @brief Get the i-th box's proc id.
   *
   * @param i given index of the box
   * @return unsigned int The proc id of the box required.
   */
  int getProcID(int i) const { return procs_.at(i); };

  /**
   * @brief Get the i-th box's Data index.
   *
   * @param box_id given index of the box
   * @return unsigned int The box's data index.
   */
  std::optional<int> getDataID(int box_id) const;

  /**
   * @brief Get the number of boxes owned by the i-th proc.
   *
   * @param proc_id Given proc id
   * @return unsigned the number of boxes owned by the i-th proc.
   */
  int numBoxes(int proc_id = ProcID(MPI_COMM_WORLD)) const {
    return std::count(procs_.begin(), procs_.end(), proc_id);
  }

  bool operator==(const DisjointBoxLayout<Dim> &rhs) const {
    if (size() != rhs.size())
      return false;
    for (auto it = begin(); it.ok(); ++it)
      if (*it != rhs.getBox(it.index()))
        return false;
    return true;
  }

  template <int SDim>
  friend std::ostream &operator<<(std::ostream &,
                                  const DisjointBoxLayout<SDim> &);

protected:
  /**
   * @brief Check whether the given boxes are disjoint with each other
   *
   */
  void checkDisjoint() const;

  /**
   * @brief Sort the boxes by the left-lower coner
   *
   */
  void sort();
};

template <int Dim>
std::ostream &operator<<(std::ostream &os, const DisjointBoxLayout<Dim> &dbl) {
  os << "DisjointBoxLayout: " << std::endl;
  for (auto it = dbl.begin(); it.ok(); ++it) {
    os << "box " << it.index() << ": " << (*it);
    if (!dbl.procs_.empty())
      os << ", proc " << dbl.getProcID(it.index());
    os << std::endl;
  }
  return os;
}