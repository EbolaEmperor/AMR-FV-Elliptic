#include "FiniteDiff/MGIntergrid.h"

#include "AMRTools/Utilities.h"

template <int Dim>
template <typename T>
void MGIntergrid<Dim>::checkValidation(
    const LevelData<T, Dim> &coarseData,
    const LevelData<T, Dim> &fineData) const {
#ifndef NDEBUG
  assert(coarseData.size() == fineData.size());
  assert(coarseData.getnComps() == fineData.getnComps());
  const int nComps = coarseData.getnComps();
  for (int comp = 0; comp < nComps; ++comp) {
    assert(coarseData.getCentering(comp) == fineData.getCentering(comp));
  }
  DisjointBoxLayout<Dim> refined = coarseData.getMesh().getRefined(2);
  assert(refined == fineData.getMesh());
#endif
}

template <>
template <>
void MGIntergrid<2>::applyRestrict(const LD &fineData, LD &coarseData) const {
  UnitTimer::getInstance().begin("MGRestrict");
  checkValidation(coarseData, fineData);

  // Only support Cell-Center now...
  const int nComps = coarseData.getnComps();
  for (int comp = 0; comp < nComps; ++comp) {
    assert(coarseData.getCentering(comp) == CellCenter);
  }

  auto f_it = fineData.const_begin();
  for (auto it = coarseData.begin(); it.ok(); ++it, ++f_it) {
    const auto &cBox = it.getValidBox();

    for (int comp = 0; comp < nComps; ++comp) {
      auto &cData = it.getData()[comp];
      const auto &fData = f_it.getData()[comp];

      loop_box_2(cBox, i0, i1) {
        cData(i0, i1) =
            0.25 * (fData(2 * i0, 2 * i1) + fData(2 * i0, 2 * i1 + 1) +
                    fData(2 * i0 + 1, 2 * i1) + fData(2 * i0 + 1, 2 * i1 + 1));
      }
    }
  }
  UnitTimer::getInstance().end("MGRestrict");
}

template <>
template <>
void MGIntergrid<2>::applyRestrict(const BLD &fineData,
                                   BLD &coarseData) const {
  UnitTimer::getInstance().begin("MGRestrict");
  checkValidation(coarseData, fineData);
  const int nComps = coarseData.getnComps();
  for (int comp = 0; comp < nComps; ++comp) {
    assert(coarseData.getCentering(comp) == CellCenter);
  }
  auto f_it = fineData.const_begin();
  for (auto it = coarseData.begin(); it.ok(); ++it, ++f_it) {
    const auto &cBox = it.getValidBox();
    for (int comp = 0; comp < nComps; ++comp) {
      auto &cData = it.getData()[comp];
      const auto &fData = f_it.getData()[comp];
      loop_box_2(cBox, i0, i1) {
        cData(i0, i1) = fData(2 * i0, 2 * i1) | fData(2 * i0, 2 * i1 + 1) |
                        fData(2 * i0 + 1, 2 * i1) |
                        fData(2 * i0 + 1, 2 * i1 + 1);
      }
    }
  }
  UnitTimer::getInstance().end("MGRestrict");
}

template <>
template <typename T>
void MGIntergrid<2>::applyInterpolation(const LevelData<T, 2> &coarseData,
                                        LevelData<T, 2> &fineData) const {
  UnitTimer::getInstance().begin("MGInterp");
  checkValidation(coarseData, fineData);

  // Only support Cell-Center now...
  const int nComps = coarseData.getnComps();
  for (int comp = 0; comp < nComps; ++comp) {
    assert(coarseData.getCentering(comp) == CellCenter);
  }

  auto c_it = coarseData.const_begin();
  for (auto it = fineData.begin(); it.ok(); ++it, ++c_it) {
    const auto &cBox = c_it.getValidBox();

    for (int comp = 0; comp < nComps; ++comp) {
      auto &fData = it.getData()[comp];
      const auto &cData = c_it.getData()[comp];

      loop_box_2(cBox, i0, i1) {
        fData(2 * i0, 2 * i1) = cData(i0, i1);
        fData(2 * i0, 2 * i1 + 1) = cData(i0, i1);
        fData(2 * i0 + 1, 2 * i1) = cData(i0, i1);
        fData(2 * i0 + 1, 2 * i1 + 1) = cData(i0, i1);
      }
    }
  }
  UnitTimer::getInstance().end("MGInterp");
}

template class MGIntergrid<2>;

template void MGIntergrid<2>::applyInterpolation(const LD &coarseData,
                                                 LD &fineData) const;

template void MGIntergrid<2>::applyInterpolation(const BLD &coarseData,
                                                 BLD &fineData) const;

template void MGIntergrid<2>::applyInterpolation(
    const LevelData<int, 2> &coarseData,
    LevelData<int, 2> &fineData) const;