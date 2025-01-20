/**
 * @file LevelGhostBoxes.h
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 * @copyright Copyright (c) 2024 Wenchong Huang
 *
 */

#pragma once

#include "AMRTools/DisjointBoxLayout.h"
#include "AMRTools/ProblemDomain.h"

#include <optional>

template <int Dim>
class LevelGhostBoxes {
public:
  template <typename U>
  using Vector = std::vector<U>;
  using rVec = Vec<Real, Dim>;
  using iVec = Vec<int, Dim>;

protected:
  // Problem domain of this level
  ProblemDomain<Dim> domain_;

  // AMR mesh of this level
  DisjointBoxLayout<Dim> mesh_;

  // Ghost bandwidth of this level
  int nGhost_;

  // Ghost boxes of this level
  Vector<Box<Dim>> ghostBoxes_;

  // Which box does each ghost box belongs to
  Vector<int> belongs_;

  // Which box does each ghost box be located at.
  // Use std::nullopt to represent a ghost box located
  //   out of the mesh of this level.
  Vector<std::optional<int>> location_;

  // The i-th ghost box is the boundary of which domain.
  // If the ghost box is intergrid, fill with nullopt.
  Vector<std::optional<int>> bdryOfWhichDomain_;

  // The i-th ghost box is the boundary of which face of the domain.
  // If the ghost box is same level intergrid, fill with nullopt.
  Vector<std::optional<int>> bdryOfWhichFace_;

public:
  DisjointBoxLayout<Dim>::Iterator begin() const {
    return typename DisjointBoxLayout<Dim>::Iterator(ghostBoxes_);
  }

public:
  LevelGhostBoxes() {}

  /**
   * @brief Construct an interface class for a level of AMR.
   * @param mesh: The mesh of one level in AMR.
   * @param domain: The domain of the corresponding level.
   */
  LevelGhostBoxes(const ProblemDomain<Dim> &domain,
                  const DisjointBoxLayout<Dim> &mesh) :
      domain_(domain), mesh_(mesh) {
    nGhost_ = domain_.getNumGhosts();
    initLevelGhostBoxes();
  }

  /**
   * @brief The number of ghost boxes.
   * @return The number of ghost boxes.
   */
  unsigned size() const { return ghostBoxes_.size(); }

  int getNumGhosts() const { return nGhost_; }

  int getBelongs(const DisjointBoxLayout<Dim>::Iterator &it) const {
    return belongs_[it.index()];
  }

  int getProcID(const DisjointBoxLayout<Dim>::Iterator &it) const {
    return mesh_.getProcID(getBelongs(it));
  }

  std::optional<int> getLocation(
      const DisjointBoxLayout<Dim>::Iterator &it) const {
    return location_[it.index()];
  }

  std::optional<int> getBdryOfWhichDomain(
      const DisjointBoxLayout<Dim>::Iterator &it) const {
    return bdryOfWhichDomain_[it.index()];
  }

  std::optional<int> getBdryOfWhichFace(
      const DisjointBoxLayout<Dim>::Iterator &it) const {
    return bdryOfWhichFace_[it.index()];
  }

protected:
  /**
   * @brief Initialize interfaces by the give mesh and domain.
   *        Only be called by the construct function.
   * @param mesh: The mesh of one level in AMR.
   * @param domain: The domain of the corresponding level.
   */
  void initLevelGhostBoxes();
};

/**
 * @brief Output all of the ghost boxes.
 * @param os: The output stream.
 * @param face: The LevelGhostBoxes class to output.
 */
template <int Dim>
std::ostream &operator<<(std::ostream &os, const LevelGhostBoxes<Dim> &lgb) {
  os << "LevelGhostBoxes (size " << lgb.size() << "):" << std::endl;
  for (auto it = lgb.begin(); it.ok(); ++it) {
    os << "  Ghost Box " << (*it) << ", belongs to " << lgb.getBelongs(it);
    if (lgb.getLocation(it).has_value())
      os << ", located at " << lgb.getLocation(it).value();
    if (lgb.getBdryOfWhichDomain(it).has_value())
      os << ", next to the boundary of domain "
         << lgb.getBdryOfWhichDomain(it).value() << ", face "
         << lgb.getBdryOfWhichFace(it).value();
    os << std::endl;
  }
  return os;
}