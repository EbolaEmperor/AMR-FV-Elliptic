#include "FiniteDiff/GhostFiller.h"

#include "AMRTools/Utilities.h"
#include "Core/TensorSlice.h"

template <>
void GhostFiller<2>::fillGhosts(LevelData<Real, 2> &aData,
                                Vector<Array<char, 4>> bcType,
                                unsigned comp) const {
  UnitTimer::getInstance().begin("fillGhosts");
  assert(mesh_ == aData.getMesh());
  for (auto it = lgb_.begin(); it.ok(); ++it) {
    auto domainID = lgb_.getBdryOfWhichDomain(it);
    if (domainID.has_value() &&
        mesh_.getProcID(lgb_.getBelongs(it)) == ProcID(MPI_COMM_WORLD)) {
      int face = lgb_.getBdryOfWhichFace(it).value();
      doFillGhosts(aData.getBoxData(lgb_.getBelongs(it))[comp],
                   face >> 1,
                   (face & 1) ? 1 : -1,
                   bcType[domainID.value()][face],
                   0.,
                   aData.getCentering(comp));
    }
  }
  UnitTimer::getInstance().end("fillGhosts");
}

template <>
void GhostFiller<2>::fillGhostsOd2(LevelData<Real, 2> &aData,
                                   Vector<Array<char, 4>> bcType,
                                   unsigned comp) const {
  UnitTimer::getInstance().begin("fillGhosts");
  assert(mesh_ == aData.getMesh());
  for (auto it = lgb_.begin(); it.ok(); ++it) {
    auto domainID = lgb_.getBdryOfWhichDomain(it);
    if (domainID.has_value() &&
        mesh_.getProcID(lgb_.getBelongs(it)) == ProcID(MPI_COMM_WORLD)) {
      int face = lgb_.getBdryOfWhichFace(it).value();
      doFillGhostsOd2(aData.getBoxData(lgb_.getBelongs(it))[comp],
                      face >> 1,
                      (face & 1) ? 1 : -1,
                      bcType[domainID.value()][face],
                      0.,
                      aData.getCentering(comp));
    }
  }
  UnitTimer::getInstance().end("fillGhosts");
}

template <>
void GhostFiller<2>::fillGhostsCFHomo(LevelData<Real, 2> &aData,
                                      unsigned comp,
                                      bool useZeroGhost) const {
  UnitTimer::getInstance().begin("fillGhostsCFHomo");
  assert(mesh_ == aData.getMesh());
  const int nG = lgb_.getNumGhosts();

  for (auto it = lgb_.begin(); it.ok(); ++it) {
    auto domainID = lgb_.getBdryOfWhichDomain(it);
    auto faceID = lgb_.getBdryOfWhichFace(it);
    if (!domainID.has_value() && faceID.has_value() &&
        mesh_.getProcID(lgb_.getBelongs(it)) == ProcID(MPI_COMM_WORLD)) {
      int face = faceID.value();
      int cent = face >> 1;

      auto &aDataComp = aData.getBoxData(lgb_.getBelongs(it))[comp];
      doFillGhosts(aDataComp,
                   cent,
                   (face & 1) ? 1 : -1,
                   useZeroGhost ? 'Z' : 'H',
                   0,
                   aData.getCentering(comp));
    }
  }
  UnitTimer::getInstance().end("fillGhostsCFHomo");
}

template <>
void GhostFiller<2>::fillGhosts(
    LevelData<Real, 2> &aData,
    const Vector<Array<Tensor<Real, 1>, 4>> &cDatas,
    Vector<Array<char, 4>> bcType,
    unsigned comp) const {
  UnitTimer::getInstance().begin("fillGhosts");
  assert(mesh_ == aData.getMesh());
  const int nG = lgb_.getNumGhosts();

  for (auto it = lgb_.begin(); it.ok(); ++it) {
    auto domainID = lgb_.getBdryOfWhichDomain(it);
    if (domainID.has_value() &&
        mesh_.getProcID(lgb_.getBelongs(it)) == ProcID(MPI_COMM_WORLD)) {
      int face = lgb_.getBdryOfWhichFace(it).value();
      int cent = face >> 1;

      auto &aDataComp = aData.getBoxData(lgb_.getBelongs(it))[comp];
      auto &cData = cDatas[domainID.value()][face];

      Box<2> data_box = aDataComp.box().grow(-nG);
      auto cDataSlice =
          (face & 1)
              ? cData.slice(reduce(
                    data_box.highSideBox(cent).grow(nG, cent ^ 1), cent))
              : cData.slice(reduce(
                    data_box.lowSideBox(cent).grow(nG, cent ^ 1), cent));

      doFillGhosts(aDataComp,
                   cent,
                   (face & 1) ? 1 : -1,
                   bcType[domainID.value()][face],
                   cDataSlice,
                   aData.getCentering(comp));
    }
  }
  UnitTimer::getInstance().end("fillGhosts");
}

template <int Dim>
template <typename T>
void GhostFiller<Dim>::doFillGhosts(Tensor<Real, Dim> &aData,
                                    int D,
                                    int side,
                                    char type,
                                    const T &cData,
                                    int centering) const {
  // get essential information
  const int nG = lgb_.getNumGhosts();
  // data_box is the valid box
  Box<Dim> data_box = aData.box().grow(-nG);

  assert(nG >= 2);
  int bound[] = {data_box.lo()[D], data_box.hi()[D]};
  if (side > 0) {
    std::swap(bound[0], bound[1]);
  }
  int ito;
  if (type == 'P') {
    // TODO
  } else {
    auto dx = domain_.getDx();
    if (centering == D) {
      switch (type) {
        case 'N':
          ito = bound[0] + side;
          aData.slice(D, ito) = aData.slice(D, ito - side) * (-10.0 / 3) +
                                aData.slice(D, ito - 2 * side) * (18.0 / 3) +
                                aData.slice(D, ito - 3 * side) * (-6.0 / 3) +
                                aData.slice(D, ito - 4 * side) * (1.0 / 3) +
                                cData * (4.0 * dx[D]);  // near side
          aData.slice(D, ito + side) =
              aData.slice(D, ito - side) * (-80.0 / 3) +
              aData.slice(D, ito - 2 * side) * (120.0 / 3) +
              aData.slice(D, ito - 3 * side) * (-45.0 / 3) +
              aData.slice(D, ito - 4 * side) * (8.0 / 3) +
              cData * (20.0 * dx[D]);  // far side
          break;
        case 'D':
          aData.slice(D, bound[0]) = cData;
          // NO break !!!
        case 'X':
          ito = bound[0] + side;
          aData.slice(D, ito) = aData.slice(D, ito - side) * (5.0) +
                                aData.slice(D, ito - 2 * side) * (-10.0) +
                                aData.slice(D, ito - 3 * side) * (10.0) +
                                aData.slice(D, ito - 4 * side) * (-5.0) +
                                aData.slice(D, ito - 5 * side) * (1.0);
          break;
      }
    }  // if rd_.getCentering() == D
    else {
      ito = bound[0] + side;
      switch (type) {
        case 'N':
          aData.slice(D, ito) = aData.slice(D, ito - side) * (1.0 / 2) +
                                aData.slice(D, ito - 2 * side) * (9.0 / 10) +
                                aData.slice(D, ito - 3 * side) * (-1.0 / 2) +
                                aData.slice(D, ito - 4 * side) * (1.0 / 10) +
                                cData * (6.0 / 5 * dx[D]);  // near side
          aData.slice(D, ito + side) =
              aData.slice(D, ito - side) * (-15.0 / 2) +
              aData.slice(D, ito - 2 * side) * (29.0 / 2) +
              aData.slice(D, ito - 3 * side) * (-15.0 / 2) +
              aData.slice(D, ito - 4 * side) * (3.0 / 2) +
              cData * (6.0 * dx[D]);  // far side
          break;
        case 'D':
          aData.slice(D, ito) = aData.slice(D, ito - side) * (-77.0 / 12) +
                                aData.slice(D, ito - 2 * side) * (43.0 / 12) +
                                aData.slice(D, ito - 3 * side) * (-17.0 / 12) +
                                aData.slice(D, ito - 4 * side) * (1.0 / 4) +
                                cData * (5.0);  // near side
          aData.slice(D, ito + side) =
              aData.slice(D, ito - side) * (-505.0 / 12) +
              aData.slice(D, ito - 2 * side) * (335.0 / 12) +
              aData.slice(D, ito - 3 * side) * (-145.0 / 12) +
              aData.slice(D, ito - 4 * side) * (9.0 / 4) +
              cData * (25.0);  // far side
          break;
        case 'H':  // homogeneous C-F interpolation
          aData.slice(D, ito) =
              aData.slice(D, ito - side) * (137. / 58) +
              aData.slice(D, ito - 2 * side) * (-1307. / 522) +
              aData.slice(D, ito - 3 * side) * (673. / 522) +
              aData.slice(D, ito - 4 * side) * (-137. / 522);
          aData.slice(D, ito + side) =
              aData.slice(D, ito - side) * (105. / 58) +
              aData.slice(D, ito - 2 * side) * (-1315. / 522) +
              aData.slice(D, ito - 3 * side) * (755. / 522) +
              aData.slice(D, ito - 4 * side) * (-163. / 522);
          break;
        case 'Z':  // set ghosts to be 0 for homogeneous C-F interpolation
          aData.slice(D, ito) = 0.;
          aData.slice(D, ito + side) = 0.;
          break;
        case 'X':
          aData.slice(D, ito) = aData.slice(D, ito - side) * (5.0) +
                                aData.slice(D, ito - 2 * side) * (-10.0) +
                                aData.slice(D, ito - 3 * side) * (10.0) +
                                aData.slice(D, ito - 4 * side) * (-5.0) +
                                aData.slice(D, ito - 5 * side) * (1.0);
          aData.slice(D, ito + side) =
              aData.slice(D, ito - side) * (15.0) +
              aData.slice(D, ito - 2 * side) * (-40.0) +
              aData.slice(D, ito - 3 * side) * (45.0) +
              aData.slice(D, ito - 4 * side) * (-24.0) +
              aData.slice(D, ito - 5 * side) * (5.0);
          break;
      }
    }
  }
}

template <int Dim>
template <typename T>
void GhostFiller<Dim>::doFillGhostsOd2(Tensor<Real, Dim> &aData,
                                       int D,
                                       int side,
                                       char type,
                                       const T &cData,
                                       int centering) const {
  // get essential information
  const int nG = lgb_.getNumGhosts();
  // data_box is the valid box
  Box<Dim> data_box = aData.box().grow(-nG);

  assert(nG >= 1);
  int bound[] = {data_box.lo()[D], data_box.hi()[D]};
  if (side > 0) {
    std::swap(bound[0], bound[1]);
  }
  int ito;
  if (type == 'P') {
    // TODO
  } else {
    auto dx = domain_.getDx();
    if (centering == D) {
      switch (type) {
        case 'N':
          ito = bound[0] + side;
          aData.slice(D, ito) = aData.slice(D, ito - 2 * side) +
                                cData * (2.0 * dx[D]);  // near side
          break;
        case 'D':
          aData.slice(D, bound[0]) = cData;
          // NO break !!!
        case 'X':
          ito = bound[0] + side;
          aData.slice(D, ito) = aData.slice(D, ito - side) * (3.0) +
                                aData.slice(D, ito - 2 * side) * (-3.0) +
                                aData.slice(D, ito - 3 * side) * (1.0);
          break;
      }
    }  // if rd_.getCentering() == D
    else {
      ito = bound[0] + side;
      switch (type) {
        case 'N':
          aData.slice(D, ito) =
              aData.slice(D, ito - side) + cData * dx[D];  // near side
          break;
        case 'D':
          aData.slice(D, ito) = aData.slice(D, ito - side) * (-2.5) +
                                aData.slice(D, ito - 2 * side) * (0.5) +
                                cData * (3.0);  // near side
          break;
        case 'Z':  // set ghosts to be 0 for homogeneous C-F interpolation
          aData.slice(D, ito) = 0.;
          aData.slice(D, ito + side) = 0.;
          break;
        case 'X':
          aData.slice(D, ito) = aData.slice(D, ito - side) * (3.0) +
                                aData.slice(D, ito - 2 * side) * (-3.0) +
                                aData.slice(D, ito - 3 * side) * (1.0);
          break;
      }
    }
  }
}

template <>
void GhostFiller<2>::fillCorners(LevelData<Real, 2> &aData) const {
  // Fill corner ghosts with 5-points regular extrapolation.
  // But I cannot ensure the performance...
  const auto &mesh = mesh_;
  const auto &domainDBL = domain_.getLayout();
  const int nComps = aData.getnComps();
  const int nG = lgb_.getNumGhosts();

  for (auto it = aData.begin(); it.ok(); ++it) {
    for (int comp = 0; comp < nComps; ++comp) {
      auto &aData = it.getData()[comp];
      auto box = aData.box();
      Vector<Box<2>> duBoxes;
      duBoxes.emplace_back(box.lo(),
                           (iVec){box.hi()[0], box.lo()[1] + nG - 1});
      duBoxes.emplace_back((iVec){box.lo()[0], box.hi()[1] - nG + 1},
                           box.hi());

      auto extrapolation = [&](const iVec &i, int side) {
        aData(i[0] + side, i[1]) = aData(i[0], i[1]) * 5.0 +
                                   aData(i[0] - side, i[1]) * (-10.0) +
                                   aData(i[0] - 2 * side, i[1]) * 10.0 +
                                   aData(i[0] - 3 * side, i[1]) * (-5.0) +
                                   aData(i[0] - 4 * side, i[1]);
        aData(i[0] + 2 * side, i[1]) = aData(i[0], i[1]) * 15.0 +
                                       aData(i[0] - side, i[1]) * (-40.0) +
                                       aData(i[0] - 2 * side, i[1]) * 45.0 +
                                       aData(i[0] - 3 * side, i[1]) * (-24.0) +
                                       aData(i[0] - 4 * side, i[1]) * 5.0;
      };

      for (auto &bBox : duBoxes) {
        if (domainDBL.contain(bBox.lo()) && !mesh.contain(bBox.lo())) {
          for (int k = 0; k < nG; k++)
            extrapolation(bBox.lo() + (iVec){2, k}, -1);
        }
        if (domainDBL.contain(bBox.hi()) && !mesh.contain(bBox.hi())) {
          for (int k = 0; k < nG; k++)
            extrapolation(bBox.hi() + (iVec){-2, -k}, 1);
        }
      }
    }
  }
}

template class GhostFiller<2>;