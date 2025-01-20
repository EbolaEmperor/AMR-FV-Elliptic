#include "AMRTools/AMRMeshHierachy.h"

template <int Dim>
unsigned AMRMeshHierachy<Dim>::numBoxes(int proc_id) const {
  unsigned all = 0;
  for (auto &mesh : meshes_) {
    if (proc_id == -1)
      all += mesh.size();
    else
      all += mesh.numBoxes(proc_id);
  }
  return all;
}

template <int Dim>
void AMRMeshHierachy<Dim>::initAll() {
  parents_.resize(meshes_.size());
  for (unsigned i = 0; i < meshes_.size(); ++i) {
    lgb_.emplace_back(getDomain(i), getMesh(i));
    if (i)
      computeParentIndices(i);
  }
}

template <int Dim>
bool AMRMeshHierachy<Dim>::checkProcID(unsigned level) const {
  assert(level > 0);
  const auto &mesh = getMesh(level);
  const auto &paMesh = getMesh(level - 1);
  for (auto it = getMesh(level).begin(); it.ok(); ++it) {
    if (mesh.getProcID(it.index()) !=
        paMesh.getProcID(getParent(level, it.index())))
      return false;
  }
  return true;
}

template <int Dim>
void AMRMeshHierachy<Dim>::computeParentIndices(unsigned level) {
  assert(level > 0);
  const auto &paMesh = getMesh(level - 1);
  const auto &mesh = getMesh(level);
  const auto &refRatio = getRefRatioToNextLevel(level - 1);
  auto coMesh = mesh.getCoarsened(refRatio);
  auto &paIDs = parents_[level];

  paIDs.resize(mesh.size());
  for (auto it = mesh.begin(); it.ok(); ++it) {
    auto paID = paMesh.whichBox(it->lo() / refRatio);

    // Check whether the fine box totally lies in a coarse box.
    if (!paID.has_value() ||
        !paMesh.getBox(paID.value()).contain(coMesh.getBox(it.index()))) {
      std::stringstream errmsg;
      errmsg << "Invalid mesh in level " << level;
      throw std::runtime_error(errmsg.str());
    }

    paIDs[it.index()] = paID.value();
  }

  if (!checkProcID(level)) {
    throw std::runtime_error("Invalid process assignment!");
  }
}

template <int Dim>
void AMRMeshHierachy<Dim>::push_back(const DisjointBoxLayout<Dim> &newMesh,
                                     int refRatio) {
  meshes_.push_back(newMesh);
  domains_.push_back(domains_.back().getRefined(refRatio));
  lgb_.emplace_back(domains_.back(), meshes_.back());
  refRatios_.push_back(refRatio);
  parents_.resize(size());
  computeParentIndices(size() - 1);
}

template class AMRMeshHierachy<2>;