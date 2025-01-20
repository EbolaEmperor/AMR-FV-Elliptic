#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::computeCurl(const LD &u, LD &curl) const {
  if (u.getnComps() != 2 || u.getCentering(0) != FaceCenter0 ||
      u.getCentering(1) != FaceCenter1) {
    throw std::runtime_error(
        "LevelOp:: Curl(u) can only be applied on a face-centered vector.");
  }
  if (curl.getnComps() != 1 || curl.getCentering(0) != NodeCenter) {
    throw std::runtime_error("LevelOp<2>:: The result of Curl(u) should be a "
                             "node-centered scalar.");
  }
  auto dx = domain_.getDx();

  auto srcit = u.const_begin();
  for (auto dstit = curl.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &u0 = srcit.getData()[0];
    const auto &u1 = srcit.getData()[1];
    auto box = dstit.getValidBox(0);
    auto &v = dstit.getData()[0];
    loop_box_2(box, i, j) {
      v(i, j) = (-u1(i + 1, j) + u1(i, j) * 15.0 - u1(i - 1, j) * 15.0 +
                 u1(i - 2, j)) /
                    (12. * dx[0]) +
                -(-u0(i, j + 1) + u0(i, j) * 15.0 - u0(i, j - 1) * 15.0 +
                  u0(i, j - 2)) /
                    (12.0 * dx[1]);
    }
  }
}

template class LevelOp<2>;