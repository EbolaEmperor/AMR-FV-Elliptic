#include "../example/common/DomainFactory.cpp"
#include "catch_amalgamated.hpp"

#include <AMRTools/AMRMeshHierachy.h>
#include <AMRTools/BRMeshRefine.h>
#include <AMRTools/LevelData.h>
#include <AMRTools/ProblemDomain.h>
#include <bit>
#include <cmath>
// Using GaussLegendre to fill the control volumes.
#include "Core/numlib.h"
// Using computeNorm to get the bound of tags.
#include "FiniteDiff/LevelOp.h"

using namespace std;
using iVec = Vec<int, 2>;
using rVec = Vec<double, 2>;

template <typename TFunc>
void filltestData(LevelData<Real, 2> &Data_,
                  const AMRMeshHierachy<2> &amrHier,
                  const TFunc &expr);

void computetestTags(const LevelData<Real, 2> &grad,
                     const AMRMeshHierachy<2> &amrHier,
                     LevelData<bool, 2> &tags,
                     Real thereshold);
LevelData<bool, 2> ProduceTags_bool(const AMRMeshHierachy<2> &amrHier,
                                    Real thereshold);

template <class T, class Checker>
bool checkMeshRefine(const DisjointBoxLayout<2> ref_mesh,
                     const T &Size_value,
                     const Checker &checker) {
  for (auto it = ref_mesh.begin(); it.ok(); ++it) {
    if (!checker(it->size()[0], Size_value) ||
        !checker(it->size()[1], Size_value))
      return false;
  }
  return true;
};

TEST_CASE("2-D BRMeshRefine Test for UnitSquare", "[BRMeshRefine]") {
  int nProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  if (std::popcount((unsigned)nProcs) != 1)
    return;
  auto dwrU = NS4_Domains::UnitSquare(512);

  SECTION("UnitSquare Test for bool, \rho = 0.3") {
    auto domain = dwrU.getDomain();
    auto mesh = dwrU.getMesh();
    AMRMeshHierachy<2> amrHier(domain, mesh);
    auto tags = ProduceTags_bool(amrHier, 0.3);
    BRMeshRefine<2> refiner(amrHier.getMesh(0));
    // refiner.setFillRatio(0.5);
    auto ref_mesh = refiner.makeLayout(tags, 8);
    REQUIRE(checkMeshRefine(ref_mesh, 4, [](int a, int b) { return a >= b; }));
  }

  SECTION("UnitSquare Test for bool, \rho = 0.6") {
    auto domain = dwrU.getDomain();
    auto mesh = dwrU.getMesh();
    AMRMeshHierachy<2> amrHier(domain, mesh);
    auto tags = ProduceTags_bool(amrHier, 0.6);
    BRMeshRefine<2> refiner(amrHier.getMesh(0));
    // refiner.setFillRatio(0.5);
    auto ref_mesh = refiner.makeLayout(tags, 8);
    REQUIRE(checkMeshRefine(ref_mesh, 4, [](int a, int b) { return a >= b; }));
  }

  SECTION("UnitSquare Test for bool, \rho = 0.9") {
    auto domain = dwrU.getDomain();
    auto mesh = dwrU.getMesh();
    AMRMeshHierachy<2> amrHier(domain, mesh);
    auto tags = ProduceTags_bool(amrHier, 0.9);
    BRMeshRefine<2> refiner(amrHier.getMesh(0));
    // refiner.setFillRatio(0.5);
    auto ref_mesh = refiner.makeLayout(tags, 8);
    REQUIRE(checkMeshRefine(ref_mesh, 4, [](int a, int b) { return a >= b; }));
  }
}

TEST_CASE("2-D BRMeshRefine Test for Lshape", "[BRMeshRefine]") {
  int nProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  if (nProcs != 1 &&
      (nProcs % 3 != 0 || std::popcount((unsigned)(nProcs / 3)) != 1))
    return;
  auto dwrL = NS4_Domains::Lshape(512);

  SECTION("Lshape Test for bool, \rho = 0.3") {
    auto domain = dwrL.getDomain();
    auto mesh = dwrL.getMesh();
    AMRMeshHierachy<2> amrHier(domain, mesh);
    auto tags = ProduceTags_bool(amrHier, 0.3);
    BRMeshRefine<2> refiner(amrHier.getMesh(0));
    // refiner.setFillRatio(0.5);
    auto ref_mesh = refiner.makeLayout(tags, 8);
    REQUIRE(checkMeshRefine(ref_mesh, 4, [](int a, int b) { return a >= b; }));
  }

  SECTION("Lshape Test for bool, \rho = 0.6") {
    auto domain = dwrL.getDomain();
    auto mesh = dwrL.getMesh();
    AMRMeshHierachy<2> amrHier(domain, mesh);
    auto tags = ProduceTags_bool(amrHier, 0.6);
    BRMeshRefine<2> refiner(amrHier.getMesh(0));
    // refiner.setFillRatio(0.5);
    auto ref_mesh = refiner.makeLayout(tags, 8);
    REQUIRE(checkMeshRefine(ref_mesh, 4, [](int a, int b) { return a >= b; }));
  }

  SECTION("Lshape Test for bool, \rho = 0.9") {
    auto domain = dwrL.getDomain();
    auto mesh = dwrL.getMesh();
    AMRMeshHierachy<2> amrHier(domain, mesh);
    auto tags = ProduceTags_bool(amrHier, 0.9);
    BRMeshRefine<2> refiner(amrHier.getMesh(0));
    // refiner.setFillRatio(0.5);
    auto ref_mesh = refiner.makeLayout(tags, 8);
    REQUIRE(checkMeshRefine(ref_mesh, 4, [](int a, int b) { return a >= b; }));
  }
};

template <typename TFunc>
void filltestData(LevelData<Real, 2> &Data_,
                  const AMRMeshHierachy<2> &amrHier,
                  const TFunc &expr) {
  UnitTimer::getInstance().begin("filltestData");
  auto domain = amrHier.getDomain(0);
  rVec dx = domain.getDx();
  rVec x0 = domain.getX0();
  auto area = dx[0] * dx[1];
  for (auto it = Data_.begin(); it.ok(); ++it) {
    auto box = it.getValidBox(0);
    auto &data = it.getData()[0];
    loop_box_2(box, i0, i1) {
      rVec iv = {i0 * dx[0] + x0[0], i1 * dx[1] + x0[1]};
      data(i0, i1) = quad<4>(expr, iv, iv + dx) / area;
    }
  }
};

void computetestTags(const LevelData<Real, 2> &grad,
                     const AMRMeshHierachy<2> &amrHier,
                     LevelData<bool, 2> &tags,
                     Real thereshold) {
  auto LvOp = amrHier.createLevelOp(0);
  auto MaxGrad = LvOp.computeNorm(grad, 0);
  int nComps = grad.getnComps();
  assert((int)MaxGrad.size() == nComps == 1);
  Real bound = MaxGrad[0] * thereshold;
  tags.memset(0);

  auto ite_tags = tags.begin();
  auto ite_grad = grad.const_begin();
  for (; ite_grad.ok(); ++ite_tags, ++ite_grad) {
    auto &tData = ite_tags.getData()[0];
    auto box = ite_grad.getValidBox(0);
    const auto &gData = ite_grad.getData()[0];
    loop_box_2(box, i0, j0) {
      if (fabs(gData(i0, j0) >= bound))
        tData(i0, j0) = true;
    }
  }
};

LevelData<bool, 2> ProduceTags_bool(const AMRMeshHierachy<2> &amrHier,
                                    Real thereshold) {
  // Describe the region of facial of Sacaban Snapper
  const double r = 0.3;
  auto x1 = (rVec){0.5 + r / 2 * sqrt(3), 0.5 + r / 2};
  auto x2 = (rVec){0.5 - r / 2 * sqrt(3), 0.5 + r / 2};
  auto tr1 = (rVec){0.4, 0.55};
  auto tr2 = (rVec){0.6, 0.55};
  auto tr3 = (rVec){0.5, 0.45};

  auto cross = [](const rVec &a, const rVec &b) {
    return a[0] * b[1] - a[1] * b[0];
  };

  auto f = [&](const rVec &x) {
    auto d = norm(x - x1);
    if (d < 0.1 || (d >= 0.12 && d <= 0.13))
      return 1;
    d = norm(x - x2);
    if (d < 0.1 || (d >= 0.12 && d <= 0.13))
      return 1;
    if (cross(tr3 - tr1, x - tr1) >= 0 && cross(x - tr1, tr2 - tr1) >= 0 &&
        cross(tr2 - tr3, x - tr3) >= 0)
      return 1;
    return 0;
  };

  // Produce the tags while the default centering is cellcenter.
  auto grad = amrHier.createLevelData<Real>(0, CellCenter);
  filltestData(grad, amrHier, f);
  auto tag = amrHier.createLevelData<bool>(0, CellCenter);
  computetestTags(grad, amrHier, tag, thereshold);
  return tag;
};