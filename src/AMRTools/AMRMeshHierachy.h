/**
 * @file AMRMeshHandler.h
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 * @copyright Copyright (c) 2024 Wenchong Huang
 *
 */

#pragma once

#include "AMRTools/DisjointBoxLayout.h"
#include "AMRTools/LevelData.h"
#include "AMRTools/LevelGhostBoxes.h"
#include "AMRTools/ProblemDomain.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "FiniteDiff/LevelOp.h"

template <int Dim>
class AMRMeshHierachy {
private:
  std::vector<DisjointBoxLayout<Dim>> meshes_;

  std::vector<LevelGhostBoxes<Dim>> lgb_;

  std::vector<ProblemDomain<Dim>> domains_;

  // refRatios_[i] is the refine ratio from level i to level i+1.
  std::vector<int> refRatios_;

  // Parent-Box: parents_[i][j] is the box id where
  // the j-th box of level i covers.
  std::vector<std::vector<int>> parents_;

  // Bandwidth of ghost layer.
  int nGhost_;

public:
  unsigned size() const { return meshes_.size(); }

  /**
   * @brief the number of boxes in a process
   * @param proc_id: the process index, use -1 to
   *        get the total number of boxes.
   */
  unsigned numBoxes(int proc_id = -1) const;

  const DisjointBoxLayout<Dim> &getMesh(unsigned level) const {
    return meshes_.at(level);
  }

  const ProblemDomain<Dim> &getDomain(unsigned level) const {
    return domains_.at(level);
  }

  const LevelGhostBoxes<Dim> &getLevelGhostBoxes(unsigned level) const {
    return lgb_.at(level);
  }

  int getRefRatioToNextLevel(unsigned level) const {
    return refRatios_.at(level);
  }

  const std::vector<int> &getParents(unsigned level) const {
    return parents_.at(level);
  }

  int getParent(unsigned level, int box_id) const {
    return parents_.at(level).at(box_id);
  }

  int getnGhost() const { return nGhost_; }

  AMRMeshHierachy<Dim> getGloballyRefined(int refRatio = 2) const {
    AMRMeshHierachy<Dim> refAMRHier;
    refAMRHier.meshes_.resize(size());
    refAMRHier.domains_.resize(size());
    refAMRHier.lgb_.resize(size());
    for (unsigned lv = 0; lv < size(); ++lv) {
      refAMRHier.meshes_[lv] = meshes_[lv].getRefined(refRatio);
      refAMRHier.domains_[lv] = domains_[lv].getRefined(refRatio);
      refAMRHier.lgb_[lv] = LevelGhostBoxes<Dim>(refAMRHier.domains_[lv],
                                                 refAMRHier.meshes_[lv]);
    }
    refAMRHier.refRatios_ = refRatios_;
    refAMRHier.parents_ = parents_;
    refAMRHier.nGhost_ = nGhost_;
    return refAMRHier;
  }

  /**
   * @brief Reset the mesh of the give level.
   *        Check if the new mesh is valid.
   *        Reinit parent-boxes ID and ghost boxes.
   * @param level: the level that needs to reset.
   * @param newMesh: the new mesh of the give level.
   *
   * @note NO IMPLEMENTATION YET.
   */
  void resetLevel(unsigned level, const DisjointBoxLayout<Dim> &newMesh);

  /**
   * @brief Add a new finest level
   *        Check if the new mesh is valid.
   *        Compute domain, parent-boxes ID and ghost boxes.
   * @param newMesh: the new mesh of the give level.
   * @param refRatio: the refine ratio from old finest level to the new
   * level.
   */
  void push_back(const DisjointBoxLayout<Dim> &newMesh, int refRatio);

  /**
   * @brief create a FuncFiller of the given level
   */
  FuncFiller<Dim> createFuncFiller(unsigned level) const {
    return FuncFiller<Dim>(getDomain(level), getMesh(level));
  }

  /**
   * @brief create a GhostFiller of the given level
   */
  GhostFiller<Dim> createGhostFiller(unsigned level) const {
    return GhostFiller<Dim>(
        getDomain(level), getMesh(level), getLevelGhostBoxes(level));
  }

  /**
   * @brief create a LevelOp of the given level
   */
  LevelOp<Dim> createLevelOp(unsigned level) const {
    return LevelOp<Dim>(getDomain(level));
  }

  /**
   * @brief create a LevelData of the given level
   */
  template <typename T>
  LevelData<T, Dim> createLevelData(unsigned level,
                                    int centering,
                                    unsigned nComps = 1) const {
    return LevelData<T, Dim>(getMesh(level), centering, nComps, nGhost_);
  }

  /**
   * @brief create a LevelData of the given level
   */
  template <typename T>
  LevelData<T, Dim> createLevelData(unsigned level,
                                    const std::vector<int> &centering,
                                    unsigned nComps = 1) const {
    return LevelData<T, Dim>(getMesh(level), centering, nComps, nGhost_);
  }

  /**
   * @brief create a AMR data
   */
  template <typename T>
  std::vector<LevelData<T, Dim>> createAMRData(
      const std::vector<int> &centering,
      unsigned nComps = 1) const {
    std::vector<LevelData<T, Dim>> amrData;
    for (unsigned lv = 0; lv < size(); ++lv)
      amrData.push_back(createLevelData<T>(lv, centering, nComps));
    return amrData;
  }

  /**
   * @brief create an AMR data
   */
  template <typename T>
  std::vector<LevelData<T, Dim>> createAMRData(int centering,
                                               unsigned nComps = 1) const {
    std::vector<int> cents(nComps, centering);
    return createAMRData<T>(cents, nComps);
  }

  std::vector<std::vector<std::array<Tensor<Real, 1>, 4>>> createAMRBdryData()
      const {
    std::vector<std::vector<std::array<Tensor<Real, 1>, 4>>> allcDatas(size());
    for (unsigned i = 0; i < size(); i++)
      allcDatas[i].resize(getDomain(i).size());
    return allcDatas;
  }

  /**
   * @brief create a LevelData of the given level,
   *        who stores a face-centered vector.
   */
  template <typename T>
  LevelData<T, Dim> createLevelDataFaceVector(unsigned level) const {
    std::vector<int> cents(Dim);
    for (int d = 0; d < Dim; ++d)
      cents[d] = d;
    return LevelData<T, Dim>(getMesh(level), cents, Dim, nGhost_);
  }

  /**
   * @brief create an AMR Data of the given level,
   *        who stores a face-centered vector.
   */
  template <typename T>
  std::vector<LevelData<T, Dim>> createAMRDataFaceVector() const {
    std::vector<LevelData<T, Dim>> amrVectorData;
    for (unsigned i = 0; i < size(); ++i)
      amrVectorData.push_back(createLevelDataFaceVector<T>(i));
    return amrVectorData;
  }

protected:
  /**
   * @brief Compute parent-boxes indices of the level.
   * @param level: the level that needs to compute.
   */
  void computeParentIndices(unsigned level);

  /**
   * @brief Check whether the boxes of the given level
   *        are assigned to the same process of
   *        corresponding parent-boxes.
   * @param level: the level that needs to check.
   */
  bool checkProcID(unsigned level) const;

  /**
   * @brief Initialize all the infomations
   */
  void initAll();

public:
  AMRMeshHierachy() {}

  /**
   * @brief Construct a new AMRMeshHandler with only a base level
   *
   * @param baseMesh: the base mesh.
   * @param baseDomain: the base domain.
   */
  AMRMeshHierachy(const ProblemDomain<Dim> &baseDomain,
                  const DisjointBoxLayout<Dim> &baseMesh) {
    domains_.push_back(baseDomain);
    meshes_.push_back(baseMesh);
    nGhost_ = baseDomain.getNumGhosts();
    initAll();
  }

  /**
   * @brief The moving constructor with only a base level
   * @param dying values
   */
  AMRMeshHierachy(ProblemDomain<Dim> &&baseDomain,
                  DisjointBoxLayout<Dim> &&baseMesh) {
    domains_.push_back(std::move(baseDomain));
    meshes_.push_back(std::move(baseMesh));
    nGhost_ = baseDomain.getNumGhosts();
    initAll();
  }

  /**
   * @brief Construct a new AMRMeshHandler with many levels
   *
   * @param meshes: the meshes of each level.
   * @param domains: the domains of each level.
   * @param refRatios: the refine ratio from level i to level i+1
   */
  AMRMeshHierachy(const std::vector<ProblemDomain<Dim>> &domains,
                  const std::vector<DisjointBoxLayout<Dim>> &meshes,
                  const std::vector<int> &refRatios) {
    assert(meshes.size() == domains.size());
    assert(refRatios.size() == meshes.size() - 1);
    meshes_ = meshes;
    domains_ = domains;
    refRatios_ = refRatios;
    nGhost_ = domains_[0].getNumGhosts();
    initAll();
  }

  /**
   * @brief The moving constructor with many levels
   * @param dying values
   */
  AMRMeshHierachy(std::vector<ProblemDomain<Dim>> &&domains,
                  std::vector<DisjointBoxLayout<Dim>> &&meshes,
                  std::vector<int> &&refRatios) {
    assert(meshes.size() == domains.size());
    assert(refRatios.size() == meshes.size() - 1);
    meshes_ = std::move(meshes);
    domains_ = std::move(domains);
    refRatios_ = std::move(refRatios);
    nGhost_ = domains_[0].getNumGhosts();
    initAll();
  }
};

template <int Dim>
std::ostream &operator<<(std::ostream &os,
                         const AMRMeshHierachy<Dim> &amrHier) {
  os << "AMR Hierachy (" << amrHier.size() << " levels):" << std::endl;
  for (unsigned lv = 0; lv < amrHier.size(); lv++) {
    os << std::endl << " Level " << lv << ":" << std::endl;
    os << amrHier.getDomain(lv);
    os << amrHier.getMesh(lv);
    os << amrHier.getLevelGhostBoxes(lv);
    os << "Parent-box indices: ";
    for (const auto &p : amrHier.getParents(lv))
      os << p << " ";
    os << std::endl;
  }
  return os;
}