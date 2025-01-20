#include <AMRTools/AMRMeshRefine.h>
#include <AMRTools/Utilities.h>
#include <Core/LinearizationHelper.h>
#include <FiniteDiff/LevelOp.h>
#include <FiniteDiff/MGIntergrid.h>
#include <utility>

template <int Dim>
void AMRMeshRefine<Dim>::sendBoxesParallel(const Vector<Box<Dim>> &ref_boxes) {
  std::swap(const_cast<Vector<Box<Dim>> &>(ref_boxes), res_boxes_);
  BaseClass::beginAllReduce();
  // swap back to ref_boxes
  std::swap(res_boxes_, const_cast<Vector<Box<Dim>> &>(ref_boxes));
};

template <int Dim>
void AMRMeshRefine<Dim>::receiveBoxesParallel(Vector<Box<Dim>> &ref_boxes,
                                              std::vector<int> &procs) {
  BaseClass::endCommunication();
  std::swap(ref_boxes, res_boxes_);
  res_boxes_.clear();
  std::swap(procs, res_procs_);
  res_procs_.clear();
};

template <int Dim>
void AMRMeshRefine<Dim>::linearIn() {
  int proc_id = ProcID(comm_);
  BaseClass::send_buffer_.clear();
  auto &buffer = BaseClass::send_buffer_[proc_id];
  LinearizationHelper::linearIntoNewBuf(proc_id, &buffer);
  LinearizationHelper::linearIntoNewBuf(res_boxes_, &buffer);
};

template <int Dim>
void AMRMeshRefine<Dim>::linearOut() {
  auto &buffer = BaseClass::recv_buffer_[ProcID(comm_)];
  size_t pos = 0;
  res_boxes_.clear();
  res_procs_.clear();
  std::vector<Box<Dim>> tmp_boxes;
  while (pos != buffer.size()) {
    int proc_id;
    LinearizationHelper::linearOut(buffer, &pos, &proc_id);
    LinearizationHelper::linearOut(buffer, &pos, &tmp_boxes);
    res_procs_.insert(res_procs_.end(), tmp_boxes.size(), proc_id);
    res_boxes_.insert(res_boxes_.end(), tmp_boxes.begin(), tmp_boxes.end());
  }
};

template <int Dim>
DisjointBoxLayout<Dim> AMRMeshRefine<Dim>::makeLayout(
    const LevelData<bool, Dim> &tags,
    int ref_ratio) {
  UnitTimer::getInstance().begin("AMRMeshRefine");
  std::vector<Box<Dim>> ref_boxes;
  auto elgTags = enlargeTags(tags);
  makeBoxesParallel(elgTags, ref_boxes);
  std::vector<int> procs;
  sendBoxesParallel(ref_boxes);
  receiveBoxesParallel(ref_boxes, procs);
  auto refinedDBL = DisjointBoxLayout(std::move(ref_boxes), std::move(procs))
                        .getRefined(ref_ratio);
  UnitTimer::getInstance().end("AMRMeshRefine");
  return refinedDBL;
}

template <>
LevelData<bool, 2> AMRMeshRefine<2>::enlargeTags(
    const LevelData<bool, 2> &tags,
    int ratio) const {
  MGIntergrid<2> mgOp;
  Vector<LevelData<bool, 2>> levelTags;
  levelTags.push_back(tags);
  for (int r = ratio, i = 0; r >= 2; r >>= 1, ++i) {
    levelTags.push_back(
        LevelData<bool, 2>(mesh_.getCoarsened(ratio), CellCenter));
    mgOp.applyRestrict(levelTags[i], levelTags[i + 1]);
  }
  for (int i = levelTags.size() - 2; i >= 0; --i)
    mgOp.applyInterpolation(levelTags[i + 1], levelTags[i]);
  auto elgTag = levelTags[0];
  return elgTag;
}

template class AMRMeshRefine<2>;