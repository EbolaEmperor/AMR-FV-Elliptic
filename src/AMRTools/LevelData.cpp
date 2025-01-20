/**
 * @file LevelData.cpp
 * @author Jiatu Yan
 *
 * @copyright Copyright (c) 2024 Jiatu Yan
 *
 */

#include "AMRTools/LevelData.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>
#include <Core/TensorSlice.h>
#include <cstring>

//================================================ move ctor and move
template <class T, int Dim>
LevelData<T, Dim> &LevelData<T, Dim>::operator=(const T a) {
  for (auto data_itr = begin(); data_itr.ok(); ++data_itr) {
    for (unsigned i = 0; i != nComps_; ++i) {
      data_itr.getData()[i] = a;
    }
  }
  return *this;
};

template <class T, int Dim>
LevelData<T, Dim> &LevelData<T, Dim>::operator=(const LevelData<T, Dim> &rhs) {
  assert(rhs.mesh_ == mesh_);
  assert(rhs.nComps_ == nComps_);
  auto res_itr = begin();
  auto src_itr = rhs.const_begin();
  for (; res_itr.ok(); ++res_itr, ++src_itr) {
    for (size_t i = 0; i != nComps_; ++i) {
      res_itr.getData()[i] = src_itr.getData()[i];
    }
  }
  return *this;
};

// assignment
// template <class T, int Dim>
// LevelData<T, Dim>::LevelData(LevelData<T, Dim> &&rhs) noexcept {
//   std::swap(*this, rhs);
// };

// template <class T, int Dim>
// LevelData<T, Dim> &LevelData<T, Dim>::operator=(
//     LevelData<T, Dim> &&rhs) noexcept {
//   std::swap(*this, rhs);
//   return *this;
// };

template <typename T, int Dim>
void LevelData<T, Dim>::allocDatas() {
  int procID = ProcID(BaseClass::comm_);
  index_.resize(mesh_.numBoxes(procID));
  data_.resize(index_.size());
  int dataID = 0;
  for (auto iter = mesh_.begin(); iter.ok(); ++iter) {
    if (mesh_.getProcID(iter.index()) == procID) {
      index_[dataID] = iter.index();
      Box<Dim> ghostedBox(iter->lo() - nGhost_, iter->hi() + nGhost_);
      for (unsigned comp = 0; comp < nComps_; ++comp) {
        data_[dataID].emplace_back(
            staggerFromCellCenter(ghostedBox, centering_[comp]));
      }
      dataID++;
    }
  }
}

template <typename T, int Dim>
void LevelData<T, Dim>::exchangeAll() {
  UnitTimer::getInstance().begin("exchangeAll");
  BaseClass::beginCommunication();
  BaseClass::endCommunication();
  UnitTimer::getInstance().end("exchangeAll");
};

template <typename T, int Dim>
void LevelData<T, Dim>::setBoxToSendAndRecv() {
  box_to_send_.clear();
  box_index_to_send_.clear();
  box_to_recv_.clear();
  dest_box_index_.clear();
  // get the Cell-Centered covering box.
  for (auto data_itr = begin(); data_itr.ok(); ++data_itr) {
    auto data_idx = data_itr.index();
    auto box_idx = index_.at(data_idx);
    auto non_ghost_box = mesh_.getBox(box_idx);
    auto ghost_box = non_ghost_box.inflate(nGhost_);
    for (auto box_itr = mesh_.begin(); box_itr.ok(); ++box_itr) {
      auto j = box_itr.index();
      if ((int)j == box_idx) {
        continue;
      }  // skip the box itself
      auto box_from = mesh_.getBox(j);
      auto box_to = box_from.inflate(nGhost_);
      auto cover_box_to_recv = ghost_box & box_from;
      auto cover_box_to_send = non_ghost_box & box_to;
      if (!cover_box_to_recv.empty()) {
        unsigned sender_id = mesh_.getProcID(j);
        box_to_recv_[sender_id].push_back(cover_box_to_recv);
      }
      if (!cover_box_to_send.empty()) {
        unsigned receiver_id = mesh_.getProcID(static_cast<int>(j));
        box_to_send_[receiver_id].push_back(cover_box_to_send);
        box_index_to_send_[receiver_id].push_back(data_idx);
        dest_box_index_[receiver_id].push_back(j);
      }
    }  /// loop each box in the mesh
  }
};

template <typename T, int Dim>
void LevelData<T, Dim>::setDataLengthInfo() {
  BaseClass::data_length_to_send_.clear();
  BaseClass::data_length_to_recv_.clear();
  /**
   * If the data length of T is not fixed,
   * initialize the sending data length directly from send_buffer_.
   */
  if constexpr (!FixedSizeData<T>::value) {
    for (auto &buf : BaseClass::send_buffer_) {
      BaseClass::data_length_to_send_[buf.first] = buf.second.size();
    }
  } else {
    // set data length
    for (unsigned comp = 0; comp != nComps_; ++comp) {
      int centering = centering_[comp];
      auto set_length =
          [&](std::map<int, size_t> &length_to_set,
              const std::map<int, std::vector<Box<Dim>>> &box_info) {
            for (const auto &boxes : box_info) {
              auto &length = length_to_set[boxes.first];
              length += sizeof(size_t) + sizeof(int) * boxes.second.size() +
                        4 * sizeof(int) * boxes.second.size();
              for (const auto &box : boxes.second) {
                auto staggered_box = staggerFromCellCenter(box, centering);
                length += sizeof(T) * staggered_box.volume();
              }
            }  /// loop box_to_send/recv
          };
      set_length(BaseClass::data_length_to_send_, box_to_send_);
      set_length(BaseClass::data_length_to_recv_, box_to_recv_);
    }  /// loop each comp
    // set buffer_
    auto set_buffer = [&](std::map<int, std::vector<char>> &buffer,
                          const std::map<int, size_t> &data_length) {
      buffer.clear();
      for (auto &data : data_length) {
        buffer[data.first].resize(data.second);
      }
    };
    set_buffer(BaseClass::send_buffer_, BaseClass::data_length_to_send_);
    set_buffer(BaseClass::recv_buffer_, BaseClass::data_length_to_recv_);
    BaseClass::send_requests_.resize(BaseClass::data_length_to_send_.size());
    BaseClass::recv_requests_.resize(BaseClass::data_length_to_recv_.size());
  }  // T is fixed size.
};

template <typename T, int Dim>
void LevelData<T, Dim>::linearIn() {
  if constexpr (!FixedSizeData<T>::value) {
    BaseClass::send_buffer_.clear();
  }
  std::map<unsigned, size_t> positions;
  for (unsigned comp = 0; comp != nComps_; ++comp) {
    int centering = centering_[comp];
    for (const auto &itr : box_to_send_) {
      auto &pos = positions[itr.first];
      const auto &resource_index = box_index_to_send_.at(itr.first);
      auto &send_buf = BaseClass::send_buffer_[itr.first];
      auto &dest_box_index = dest_box_index_.at(itr.first);
      // number of boxes.
      if constexpr (FixedSizeData<T>::value) {
        LinearizationHelper::linearIntoOldBuf(
            resource_index.size(), &send_buf, &pos);
      } else {
        LinearizationHelper::linearIntoNewBuf(resource_index.size(),
                                              &send_buf);
      }
      for (size_t cnt = 0; cnt != resource_index.size(); ++cnt) {
        auto data_idx = resource_index[cnt];
        auto box = staggerFromCellCenter(itr.second[cnt], centering);
        auto dest_idx = dest_box_index[cnt];
        if constexpr (FixedSizeData<T>::value) {
          // dest box index info
          LinearizationHelper::linearIntoOldBuf(dest_idx, &send_buf, &pos);
          // box info
          LinearizationHelper::linearIntoOldBuf(box, &send_buf, &pos);
          // data
          LinearizationHelper::linearIntoOldBuf(
              data_[data_idx][comp].slice(box), &send_buf, &pos);
        } else {
          LinearizationHelper::linearIntoNewBuf(dest_idx, &send_buf);
          LinearizationHelper::linearIntoNewBuf(box, &send_buf);
          LinearizationHelper::linearIntoNewBuf(
              data_[data_idx][comp].slice(box), &send_buf);
        }
      }  /// loop boxes to send
    }    /// loop each processors to send.
  }      /// loop each comp
};

template <typename T, int Dim>
void LevelData<T, Dim>::linearOut() {
  std::map<int, size_t> positions;
  for (auto &recv_itr : BaseClass::recv_buffer_) {
    auto &pos = positions[recv_itr.first];
    auto &recv_buf = (recv_itr.first == ProcID(BaseClass::comm_))
                         ? BaseClass::send_buffer_[recv_itr.first]
                         : recv_itr.second;
    for (unsigned comp = 0; comp != nComps_; ++comp) {
      size_t num_boxes_to_recv;
      LinearizationHelper::linearOut(recv_buf, &pos, &num_boxes_to_recv);
      for (size_t cnt = 0; cnt != num_boxes_to_recv; ++cnt) {
        int dest_idx;
        Box<Dim> box;
        LinearizationHelper::linearOut(recv_buf, &pos, &dest_idx);
        LinearizationHelper::linearOut(recv_buf, &pos, &box);
        auto &data = getBoxData(dest_idx);
        auto sliced_data = data[comp].slice(box);
        LinearizationHelper::linearOut(recv_buf, &pos, &sliced_data);
      }  /// loop the processor that received data from
    }    /// loop each comp.
  }      /// loop the processor that received data from
};

template <typename T, int Dim>
void LevelData<T, Dim>::memset(int c) {
  for (auto it = begin(); it.ok(); ++it) {
    for (unsigned comp = 0; comp < nComps_; ++comp) {
      std::memset((void *)it.getData()[comp].data(),
                  c,
                  sizeof(T) * it.getData()[comp].box().volume());
    }
  }
}

template class LevelData<Real, 2>;
template class LevelData<int, 2>;
template class LevelData<bool, 2>;