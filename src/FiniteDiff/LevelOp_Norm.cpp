#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>
#include <Core/VecCompare.h>
#include <Core/Wrapper_OpenMP.h>
#include <Core/numlib.h>
#include <set>

template <int Dim>
std::vector<Real> LevelOp<Dim>::handleOverlappingInNorm(
    const LevelData<Real, Dim> &aData,
    int q) const {
  assert(0 <= q && q <= 2);
  auto centering = aData.getCentering();
  size_t num_comps = aData.getnComps();
  std::vector<Real> shift(num_comps, 0.0);
  for (auto data_itr = aData.const_begin(); data_itr.ok(); ++data_itr) {
    unsigned box_idx = data_itr.getBoxID();
    auto ng_cc_box = aData.getMesh().getBox(box_idx);
    const auto &data = data_itr.getData();
    for (size_t comp = 0; comp != num_comps; ++comp) {
      if (q == 0 || centering[comp] == CellCenter) {
        continue;  /// no need to handle overlapping
      }
      auto staggered_box = staggerFromCellCenter(ng_cc_box, centering[comp]);
      std::set<Vec<int, Dim>, VecCompare<int, Dim>> overlap_idx;
      for (auto box_itr = aData.getMesh().begin(); box_itr.ok(); ++box_itr) {
        if (box_itr.index() == box_idx) {
          continue;  /// skip.
        }
        auto staggered_adj_box =
            staggerFromCellCenter(*box_itr, centering[comp]);
        auto overlap_box = staggered_box & staggered_adj_box;
        if (!overlap_box.empty() && box_idx > box_itr.index()) {
          if constexpr (Dim == 2) {
            loop_box_2(overlap_box, i, j) {
              overlap_idx.insert(Vec<int, Dim>{i, j});
            }
          } else if constexpr (Dim == 3) {
            loop_box_3(overlap_box, i, j, k) {
              overlap_idx.insert(Vec<int, Dim>{i, j});
            }
          }

        }  /// handle overlap
      }    /// loop each boxes to find overlapping.
      for (auto idx : overlap_idx) {
        if (q == 1) {
          shift[comp] += std::fabs(data[comp](idx));
        } else {
          shift[comp] += pow(data[comp](idx), 2);
        }
      }
    }  /// loop for each comp;
  }    /// loop for each data boxes.
  return shift;
};

template <int Dim>
std::vector<Real> LevelOp<Dim>::computeNorm(const LevelData<Real, Dim> &aData,
                                            int q) const {
  assert(0 <= q && q <= 2);
  auto dx = domain_.getDx();
  auto centering = aData.getCentering();
  std::vector<Real> tmp(aData.getnComps(), 0.);
  std::vector<Real> res(aData.getnComps(), 0.);
  std::vector<Real> shift = handleOverlappingInNorm(aData, q);
  for (auto data_itr = aData.const_begin(); data_itr.ok(); ++data_itr) {
    /// find overlapping faces, the overlapping face on lower side is omitted.
    const auto &data = data_itr.getData();
    for (size_t comp = 0; comp != aData.getnComps(); ++comp) {
      auto data_box = data_itr.getValidBox(comp);
      if (q == 0) {
        tmp[comp] = std::max(norm(data[comp].slice(data_box), 0), tmp[comp]);
      } else if (q == 1) {
        tmp[comp] += norm(data[comp].slice(data_box), 1);
      } else {
        tmp[comp] += pow(norm(data[comp].slice(data_box), 2), 2);
      }
    }  /// loop each components
  }    /// loop each data boxes
  for (size_t comp = 0; comp != aData.getnComps(); ++comp) {
    tmp[comp] -= shift[comp];  // remove overlapping values.
    if (q != 0) {
      tmp[comp] *= prod(dx);
    }
  }
  /// sync the results
  if (q == 0) {
    MPI_Allreduce(
        tmp.data(), res.data(), res.size(), MPI_DOUBLE, MPI_MAX, comm_);
  } else {
    MPI_Allreduce(
        tmp.data(), res.data(), res.size(), MPI_DOUBLE, MPI_SUM, comm_);
  }
  if (q == 2) {
    for (auto &data : res) {
      data = sqrt(data);
    }
  }
  return res;
};

template class LevelOp<2>;