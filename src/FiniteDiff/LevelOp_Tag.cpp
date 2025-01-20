#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::computeTags(const LevelData<Real, 2> &err,
                             LevelData<bool, 2> &tags,
                             Real thereshold) const {
  auto maxerr = computeNorm(err, 0);
  int nComps = err.getnComps();
  assert((int)maxerr.size() == nComps);
  std::vector<Real> bound(nComps, 0);
  for (int comp = 0; comp < nComps; ++comp)
    bound[comp] = maxerr[comp] * thereshold;
  tags.memset(0);

  auto tagsit = tags.begin();
  auto errit = err.const_begin();
  for (; errit.ok(); ++tagsit, ++errit) {
    auto &tData = tagsit.getData()[0];

    for (int comp = 0; comp < nComps; ++comp) {
      auto box = errit.getValidBox(comp);
      const auto &aData = errit.getData()[comp];
      int cent = err.getCentering(comp);

      if (cent == CellCenter) {
        loop_box_2(box, i, j) {
          if (fabs(aData(i, j)) >= bound[comp])
            tData(i, j) = true;
        }
      } else if (cent == NodeCenter) {
        loop_box_2(box, i, j) {
          if (fabs(aData(i, j)) >= bound[comp])
            tData(i - 1, j - 1) = tData(i - 1, j) = tData(i, j - 1) =
                tData(i, j) = true;
        }
      } else if (cent == FaceCenter0) {
        loop_box_2(box, i, j) {
          if (fabs(aData(i, j)) >= bound[comp])
            tData(i - 1, j) = tData(i, j) = true;
        }
      } else {
        assert(cent == FaceCenter1);
        loop_box_2(box, i, j) {
          if (fabs(aData(i, j)) >= bound[comp])
            tData(i, j - 1) = tData(i, j) = true;
        }
      }
    }
  }
}

template class LevelOp<2>;