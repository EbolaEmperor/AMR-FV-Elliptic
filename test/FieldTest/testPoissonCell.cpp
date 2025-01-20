#include "AMRTools/LevelData.h"
#include "AMRTools/LevelDataExpr.h"
#include "AMRTools/Wrapper_Silo.h"
#include "Core/Wrapper_OpenMP.h"
#include "FiniteDiff/AMRIntergridOp.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "SpatialOp/AMRMultigrid.h"

#include <fstream>
#include <unistd.h>
#include <vector>
using namespace std;

typedef Vec<int, 2> iVec;

const int logF = 9;
const int F = 1 << logF;

AMRMeshHierachy<2> createSingleMesh() {
  vector<Box<2>> boxes;
  boxes.emplace_back((iVec){0, 0}, (iVec){7, 7});
  vector<int> procs = {0};
  DisjointBoxLayout<2> dbl(boxes, procs);
  ProblemDomain<2> pd(boxes, 1. / 8);
  return AMRMeshHierachy<2>(pd.getRefined(F), dbl.getRefined(F));
}

AMRMeshHierachy<2> createSingleProcAMRMesh() {
  vector<Box<2>> boxes;
  boxes.emplace_back((iVec){0, 0}, (iVec){7, 7});
  vector<int> procs = {0};
  DisjointBoxLayout<2> dbl(boxes, procs);
  ProblemDomain<2> pd(boxes, 1. / 8);
  AMRMeshHierachy<2> amrHier(pd.getRefined(F), dbl.getRefined(F));

  boxes[0] = Box<2>((iVec){3, 3}, (iVec){12, 12});
  DisjointBoxLayout<2> fdbl(boxes, procs);
  amrHier.push_back(fdbl.getRefined(F), 2);

  // boxes[0] = Box<2>((iVec){20, 20}, (iVec){43, 43});
  // DisjointBoxLayout<2> ffdbl(boxes, procs);
  // amrHier.push_back(ffdbl.getRefined(F / 2), 2);

  return amrHier;
}

AMRMeshHierachy<2> createMesh() {
  int nProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  assert(nProcs == 1 || nProcs == 2 || nProcs == 4);
  if (nProcs == 1)
    return createSingleProcAMRMesh();

  vector<Box<2>> boxes;
  boxes.emplace_back((iVec){3, 3}, (iVec){7, 7});
  boxes.emplace_back((iVec){3, 8}, (iVec){7, 12});
  boxes.emplace_back((iVec){8, 3}, (iVec){12, 7});
  boxes.emplace_back((iVec){8, 8}, (iVec){12, 12});
  vector<int> procs = {0, 1, 2, 3};
  if (nProcs == 2)
    procs = {0, 0, 1, 1};
  if (nProcs == 1)
    procs = {0, 0, 0, 0};

  DisjointBoxLayout<2> dbl(boxes, procs);
  dbl = dbl.getRefined(F);

  boxes.clear();
  boxes.emplace_back((iVec){0, 0}, (iVec){3, 3});
  boxes.emplace_back((iVec){0, 4}, (iVec){3, 7});
  boxes.emplace_back((iVec){4, 0}, (iVec){7, 3});
  boxes.emplace_back((iVec){4, 4}, (iVec){7, 7});
  vector<Box<2>> baseDomainBox = {Box<2>((iVec){0, 0}, (iVec){7, 7})};
  ProblemDomain<2> baseDomain(std::move(baseDomainBox), 1. / 8);
  DisjointBoxLayout<2> baseDbl(boxes, procs);
  baseDomain = baseDomain.getRefined(F);
  baseDbl = baseDbl.getRefined(F);

  boxes.clear();
  boxes.emplace_back((iVec){10, 10}, (iVec){15, 15});
  boxes.emplace_back((iVec){10, 16}, (iVec){15, 21});
  boxes.emplace_back((iVec){16, 10}, (iVec){21, 15});
  boxes.emplace_back((iVec){16, 16}, (iVec){21, 21});
  DisjointBoxLayout<2> finestDbl(boxes, procs);
  finestDbl = finestDbl.getRefined(F);

  AMRMeshHierachy<2> amrHier(baseDomain, baseDbl);
  // amrHier.push_back(dbl, 2);
  // amrHier.push_back(dbl.getRefined(2), 2);
  // amrHier.push_back(finestDbl, 2);
  return amrHier;
}

void testTagByGradient(const AMRMeshHierachy<2> &amrHier,
                       const vector<LevelData<Real, 2>> &phi,
                       Wrapper_Silo<2> &silo) {
  int nLevel = amrHier.size();
  vector<LevelData<Real, 2>> grad, gradcc, gradmag;
  vector<LevelData<bool, 2>> tags;

  for (int lv = 0; lv < nLevel; ++lv) {
    auto lvOp = amrHier.createLevelOp(lv);
    grad.push_back(amrHier.createLevelDataFaceVector<Real>(lv));
    gradcc.push_back(amrHier.createLevelData<Real>(lv, CellCenter, 2));
    gradmag.push_back(amrHier.createLevelData<Real>(lv, CellCenter));
    lvOp.computeGradient(phi[lv], grad[lv]);
    lvOp.filterFace2CellOd2(grad[lv], gradcc[lv]);
    lvOp.computeMagnitude(gradcc[lv], gradmag[lv]);
  }
  AMRIntergridOp<2> amrOp(amrHier);
  amrOp.averageToCoarse(grad);

  silo.putAMRScalar(tags, "RefineTags");
  silo.putAMRScalar(gradcc, "grad_x", 0);
  silo.putAMRScalar(gradcc, "grad_y", 1);
  silo.putAMRScalar(gradmag, "grad_magnitude");
}

void testPoisson(AMRMeshHierachy<2> amrHier) {
  AMRIntergridOp<2> amrOp(amrHier);

  vector<LevelData<Real, 2>> aDatas, realDatas, rhsDatas;
  for (unsigned i = 0; i < amrHier.size(); i++) {
    aDatas.push_back(amrHier.createLevelData<Real>(i, CellCenter));
    realDatas.push_back(amrHier.createLevelData<Real>(i, CellCenter));
    rhsDatas.push_back(amrHier.createLevelData<Real>(i, CellCenter));
  }

  const double p = 40;

  auto f = [&](const Vec<double, 2> &x) {
    return exp(-p * pow(norm(x - 0.5, 2), 2));
  };
  auto dxf = [&](const Vec<double, 2> &x) {
    return -2. * p * (x[0] - .5) * exp(-p * pow(norm(x - 0.5, 2), 2));
  };
  auto neglap = [&](const Vec<double, 2> &x) {
    return -4 * p * (p * pow(norm(x - 0.5, 2), 2) - 1) *
           exp(-p * pow(norm(x - 0.5, 2), 2));
  };

  // auto f = [&](const Vec<double, 2> &x) {
  //   return sin(M_PI * x[0]) * sin(M_PI * x[1]);
  // };
  // auto neglap = [&](const Vec<double, 2> &x) {
  //   return 2 * M_PI * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
  // };

  vector<FuncFiller<2>> func_filler;
  vector<GhostFiller<2>> ghost_filler;
  for (unsigned i = 0; i < amrHier.size(); i++) {
    func_filler.push_back(amrHier.createFuncFiller(i));
    ghost_filler.push_back(amrHier.createGhostFiller(i));
  }

  vector<array<char, 4>> bcType(1);
  for (int i = 0; i < 4; i++)
    bcType[0][i] = 'D';
  bcType[0][1] = 'N';

  vector<vector<array<Tensor<Real, 1>, 4>>> allcDatas(amrHier.size());

  for (unsigned i = 0; i < amrHier.size(); i++) {
    allcDatas[i].resize(amrHier.getDomain(i).size());
    aDatas[i].memset(0);
    func_filler[i].fillAvr(rhsDatas[i], neglap);
    func_filler[i].fillAvr(realDatas[i], f);

    func_filler[i].fillBdryAvr(allcDatas[i], f, 0, 0, FaceCenter0);
    func_filler[i].fillBdryAvr(allcDatas[i], dxf, 0, 1, FaceCenter0);
    func_filler[i].fillBdryAvr(allcDatas[i], f, 0, 2, FaceCenter1);
    func_filler[i].fillBdryAvr(allcDatas[i], f, 0, 3, FaceCenter1);
  }

  Wrapper_Silo<2> silo(".", "sol", amrHier);

  AMRMultigrid<2> multigrid(amrHier);
  multigrid.define(0., -1., CellCenter, logF + 2, 0.5);
  multigrid.setParam(6, 6, 10, 100, 1e-12, 1.01, true);
  multigrid.solve(aDatas, rhsDatas, allcDatas, bcType);

  for (unsigned i = 0; i < amrHier.size(); i++)
    realDatas[i] = realDatas[i] - aDatas[i];

  Real norm[3];
  for (int i = 0; i <= 2; ++i)
    norm[i] = amrOp.computeNorm(realDatas, i)[0];

  mpicout << "max-norm error: " << norm[0] << endl;
  mpicout << "1-norm error: " << norm[1] << endl;
  mpicout << "2-norm error: " << norm[2] << endl;

  silo.putAMRScalar(aDatas, "solution");
  silo.putAMRScalar(realDatas, "error");

  amrOp.fillGhosts(aDatas, allcDatas, bcType);
  testTagByGradient(amrHier, aDatas, silo);
}

int main() {
  MPI_Init(NULL, NULL);

  OMPSwitch omps(1);
  testPoisson(createMesh());
  UnitTimer::getInstance().report();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}