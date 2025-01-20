#include "FiniteDiff/LevelOp.h"

#include <AMRTools/LevelDataExpr.h>
#include <AMRTools/Utilities.h>

template <>
void LevelOp<2>::filterFace2Cell(const LD &u, LD &ucc) const {
  int nComps = u.getnComps();
  bool uIsFaceCenter = true;
  for (int comp = 0; comp < nComps; ++comp)
    if (u.getCentering(comp) < 0)
      uIsFaceCenter = false;
  if (!uIsFaceCenter) {
    throw std::runtime_error("LevelOp:: filterFace2Cell can only be "
                             "applied on a face-centered value.");
  }

  if (ucc.getnComps() != u.getnComps()) {
    throw std::runtime_error("LevelOp:: nComps of input and output "
                             "should be same in filterFace2Cell");
  }

  bool uccIsCellCenter = true;
  for (int comp = 0; comp < nComps; ++comp)
    if (ucc.getCentering(comp) != CellCenter)
      uccIsCellCenter = false;
  if (!uccIsCellCenter) {
    throw std::runtime_error("LevelOp:: the result of filterFace2Cell "
                             "should be a cell-centered vector");
  }

  auto srcit = u.const_begin();
  for (auto dstit = ucc.begin(); dstit.ok(); ++dstit, ++srcit) {
    for (int comp = 0; comp < nComps; ++comp) {
      const auto &uD = srcit.getData()[comp];
      auto box = dstit.getValidBox(comp);
      auto &vD = dstit.getData()[comp];
      int cent = u.getCentering(comp);
      if (cent == FaceCenter0) {
        loop_box_2(box, i, j) {
          vD(i, j) = uD(i - 1, j) * (-1.0 / 24) + uD(i, j) * (13.0 / 24) +
                     uD(i + 1, j) * (13.0 / 24) + uD(i + 2, j) * (-1.0 / 24);
        }
      } else {
        loop_box_2(box, i, j) {
          vD(i, j) = uD(i, j - 1) * (-1.0 / 24) + uD(i, j) * (13.0 / 24) +
                     uD(i, j + 1) * (13.0 / 24) + uD(i, j + 2) * (-1.0 / 24);
        }
      }
    }
  }
}

template <>
void LevelOp<2>::filterFace2CellOd2(const LD &u, LD &ucc) const {
  int nComps = u.getnComps();
  bool uIsFaceCenter = true;
  for (int comp = 0; comp < nComps; ++comp)
    if (u.getCentering(comp) < 0)
      uIsFaceCenter = false;
  if (!uIsFaceCenter) {
    throw std::runtime_error("LevelOp:: filterFace2CellOd2 can only be "
                             "applied on a face-centered value.");
  }

  if (ucc.getnComps() != u.getnComps()) {
    throw std::runtime_error("LevelOp:: nComps of input and output "
                             "should be same in filterFace2CellOd2");
  }

  bool uccIsCellCenter = true;
  for (int comp = 0; comp < nComps; ++comp)
    if (ucc.getCentering(comp) != CellCenter)
      uccIsCellCenter = false;
  if (!uccIsCellCenter) {
    throw std::runtime_error("LevelOp:: the result of filterFace2CellOd2"
                             "should be a cell-centered vector");
  }

  auto srcit = u.const_begin();
  for (auto dstit = ucc.begin(); dstit.ok(); ++dstit, ++srcit) {
    for (int comp = 0; comp < nComps; ++comp) {
      const auto &uD = srcit.getData()[comp];
      auto box = dstit.getValidBox(comp);
      auto &vD = dstit.getData()[comp];
      int cent = u.getCentering(comp);
      if (cent == FaceCenter0) {
        loop_box_2(box, i, j) vD(i, j) = .5 * (uD(i, j) + uD(i + 1, j));
      } else {
        loop_box_2(box, i, j) vD(i, j) = .5 * (uD(i, j) + uD(i, j + 1));
      }
    }
  }
}

template <>
void LevelOp<2>::filterCell2Face(const LD &u, LD &ufc) const {
  if (ufc.getnComps() != 2 || ufc.getCentering(0) != FaceCenter0 ||
      ufc.getCentering(1) != FaceCenter1) {
    throw std::runtime_error("LevelOp:: the result of filterCell2Face "
                             "should be a face-centered vector");
  }
  if (u.getnComps() != 2 || u.getCentering(0) != CellCenter ||
      u.getCentering(1) != CellCenter) {
    throw std::runtime_error("LevelOp:: filterCell2Face can only be "
                             "applied on a cell-centered vector.");
  }
  auto srcit = u.const_begin();
  for (auto dstit = ufc.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &u0 = srcit.getData()[0];
    const auto &u1 = srcit.getData()[1];
    auto box0 = dstit.getValidBox(0);
    auto box1 = dstit.getValidBox(1);
    auto &v0 = dstit.getData()[0];
    auto &v1 = dstit.getData()[1];
    loop_box_2(box0, i, j) {
      v0(i, j) = u0(i - 1, j) * (-1.0 / 12) + u0(i, j) * (7.0 / 12) +
                 u0(i + 1, j) * (7.0 / 12) + u0(i + 2, j) * (-1.0 / 12);
    }
    loop_box_2(box1, i, j) {
      v1(i, j) = u1(i, j - 1) * (-1.0 / 12) + u1(i, j) * (7.0 / 12) +
                 u1(i, j + 1) * (7.0 / 12) + u1(i, j + 2) * (-1.0 / 12);
    }
  }
}

template <>
void LevelOp<2>::filterCell2FaceOd2(const LD &u, LD &ufc) const {
  if (ufc.getnComps() != 2 || ufc.getCentering(0) != FaceCenter0 ||
      ufc.getCentering(1) != FaceCenter1) {
    throw std::runtime_error("LevelOp:: the result of filterCell2FaceOd2 "
                             "should be a face-centered vector");
  }
  if (u.getnComps() != 2 || u.getCentering(0) != CellCenter ||
      u.getCentering(1) != CellCenter) {
    throw std::runtime_error("LevelOp:: filterCell2FaceOd2 can only be "
                             "applied on a cell-centered vector.");
  }
  auto srcit = u.const_begin();
  for (auto dstit = ufc.begin(); dstit.ok(); ++dstit, ++srcit) {
    const auto &u0 = srcit.getData()[0];
    const auto &u1 = srcit.getData()[1];
    auto box0 = dstit.getValidBox(0);
    auto box1 = dstit.getValidBox(1);
    auto &v0 = dstit.getData()[0];
    auto &v1 = dstit.getData()[1];
    loop_box_2(box0, i, j) v0(i, j) = 0.5 * (u0(i - 1, j) + u0(i, j));
    loop_box_2(box1, i, j) v0(i, j) = 0.5 * (u0(i, j - 1) + u0(i, j));
  }
}

template class LevelOp<2>;