#include "AMRTools/AMRMeshHierachy.h"
#include "AMRTools/BRMeshRefine.h"
#include "AMRTools/LevelDataExpr.h"
#include "AMRTools/Wrapper_Silo.h"
#include "Core/MPI.h"
#include "Core/Wrapper_OpenMP.h"
#include "DomainFactory.h"
#include "FiniteDiff/AMRIntergridOp.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "FiniteDiff/MGIntergrid.h"
#include "FunctionFactory.h"
#include "SpatialOp/AMRMultigrid.h"

std::vector<LevelData<Real, 2>> solveAMRPoisson(
    const lightJSON::jsonNode &solverJS,
    const AMRMeshHierachy<2> &amrHier,
    int addNumBottom = 0,
    Wrapper_Silo<2> *silo = nullptr,
    bool middleOutput = false) {
  FunctionFactory<2> funFac;
  auto pRhsFunc = funFac.getFunc(solverJS["rhsFunc"]);
  auto pBdryFunc = funFac.getFunc(solverJS["boundaryFunc"]);

  int centering;
  Vec<Real, 2> HelmholtzCoef;
  std::vector<std::string> boundaryCondition;
  PGET(solverJS, HelmholtzCoef);
  PGET(solverJS, centering);
  PGET(solverJS, boundaryCondition);

  if (centering != CellCenter)
    throw std::runtime_error("Unsupported centering.");

  Vec<int, 3> relaxation;
  Real JacobiWeight;
  int maxIterations;
  Real stallThr;
  Real relRsd;
  int numBottomLevel;
  bool useFMG;
  int FMGInterpOd;

  const auto &mgJS = solverJS["multigrid"];
  PGET(mgJS, relaxation);
  PGET(mgJS, JacobiWeight);
  PGET(mgJS, maxIterations);
  PGET(mgJS, stallThr);
  PGET(mgJS, relRsd);
  PGET(mgJS, numBottomLevel);
  PGET(mgJS, useFMG);
  PGET(mgJS, FMGInterpOd);

  const auto &baseDomain = amrHier.getDomain(0);
  std::vector<std::array<char, 4>> bcType(baseDomain.size());
  for (unsigned i = 0; i < bcType.size(); ++i)
    for (int D = 0; D < 4; ++D)
      bcType[i][D] = boundaryCondition[i][D];

  std::vector<FuncFiller<2>> func_filler;
  std::vector<GhostFiller<2>> ghost_filler;
  for (unsigned i = 0; i < amrHier.size(); i++) {
    func_filler.push_back(amrHier.createFuncFiller(i));
    ghost_filler.push_back(amrHier.createGhostFiller(i));
  }

  auto aDatas = amrHier.createAMRData<Real>(centering);
  auto rhsDatas = amrHier.createAMRData<Real>(centering);
  auto allcDatas = amrHier.createAMRBdryData();

  for (unsigned i = 0; i < amrHier.size(); i++) {
    allcDatas[i].resize(amrHier.getDomain(i).size());
    aDatas[i].memset(0);
    func_filler[i].fillAvr(rhsDatas[i], *pRhsFunc);

    for (unsigned dm = 0; dm < baseDomain.size(); ++dm)
      for (int faceid = 0; faceid < 4; ++faceid) {
        if (bcType[dm][faceid] == 'D') {
          func_filler[i].fillBdryAvr(
              allcDatas[i], *pBdryFunc, dm, faceid, faceid >> 1);
        } else if (bcType[dm][faceid] == 'N') {
          Vec<Real, 2> normal = 0.;
          normal[faceid >> 1] = (faceid & 1) ? 1 : -1;
          auto normalGrad = [&](const Vec<Real, 2> &x) {
            return pBdryFunc->dot(x, normal);
          };
          func_filler[i].fillBdryAvr(
              allcDatas[i], normalGrad, dm, faceid, faceid >> 1);
        }
      }
  }

  AMRMultigrid<2> multigrid(amrHier);
  multigrid.define(HelmholtzCoef[0],
                   HelmholtzCoef[1],
                   centering,
                   numBottomLevel + addNumBottom,
                   JacobiWeight);
  multigrid.setParam(relaxation[0],
                     relaxation[1],
                     relaxation[2],
                     maxIterations,
                     relRsd,
                     stallThr,
                     useFMG,
                     FMGInterpOd);
  multigrid.solve(
      aDatas, rhsDatas, allcDatas, bcType, middleOutput ? silo : nullptr);

  if (silo != nullptr) {
    silo->putAMRScalar(aDatas, "solution");
  }
  return aDatas;
}

void Main(int argc, char *argv[]) {
  const auto *pParser = getInputParser(argv[1]);
  const auto &root = pParser->getRoot();
  const auto &grid = root["grid"];
  const auto &perf = root["perf"];
  const auto &solver = root["solver"];
  const auto &output = root["output"];

  int numThreads;
  PGET(perf, numThreads);
  OMPSwitch omps(numThreads);
  bool timing;
  PGET(perf, timing);

  int numHier;
  bool enableOutput;
  bool middleOutput;
  std::string fileName;
  PGET(grid, numHier);
  PGET(output, enableOutput);
  PGET(output, middleOutput);
  if (enableOutput)
    PGET(output, fileName);

  std::vector<int> refRatio;
  std::vector<Real> fillRatio;
  std::vector<Real> refThereshold;
  PGET(grid, refRatio);
  PGET(grid, fillRatio);
  PGET(grid, refThereshold);

  auto dwr = DomainFactory<2>::getDomainWrapper(grid);
  AMRMeshHierachy<2> amrHier0(dwr->getDomain(), dwr->getMesh());

  for (int cycle = 0; cycle < numHier - 1; ++cycle) {
    mpicout << "----------- Pre-Cycle " << cycle << " ----------" << std::endl;
    std::vector<LevelData<Real, 2>> solution;
    solution = solveAMRPoisson(solver, amrHier0);

    auto grad = amrHier0.createAMRDataFaceVector<Real>();
    auto gradcc = amrHier0.createAMRData<Real>(CellCenter, 2);
    auto gradmag = amrHier0.createAMRData<Real>(CellCenter);
    auto tags = amrHier0.createAMRData<bool>(CellCenter);
    for (int lv = 0; lv < amrHier0.size(); ++lv) {
      auto lvOp = amrHier0.createLevelOp(lv);
      lvOp.computeGradient(solution[lv], grad[lv]);
      lvOp.filterFace2CellOd2(grad[lv], gradcc[lv]);
      lvOp.computeMagnitude(gradcc[lv], gradmag[lv]);
      lvOp.computeTags(gradmag[lv], tags[lv], refThereshold[cycle]);
    }
    AMRIntergridOp<2> amrOp(amrHier0);
    amrOp.averageToCoarse(gradcc);
    amrOp.averageToCoarse(gradmag);

    DisjointBoxLayout<2> finerMesh;
    if (cycle < numHier - 1) {
      for (int lv = 0; lv < amrHier0.size(); ++lv) {
        BRMeshRefine<2> refiner(amrHier0.getMesh(lv));
        if (lv == amrHier0.size() - 1) {
          refiner.setFillRatio(fillRatio[cycle]);
          finerMesh = refiner.makeLayout(tags[lv], refRatio[cycle]);
        }
      }
      amrHier0.push_back(finerMesh, refRatio[cycle]);
    }
  }

  // now globally refine amrHier for numCycle times.
  int numCycles;
  PGET(grid, numCycles);
  std::vector<AMRMeshHierachy<2>> amrHierCycles(numCycles);
  amrHierCycles[0] = amrHier0;
  for (int cycle = 1; cycle < numCycles; ++cycle) {
    amrHierCycles[cycle] = amrHierCycles[cycle - 1].getGloballyRefined(2);
  }

  std::vector<std::vector<LevelData<Real, 2>>> solCycles(numCycles);
  for (int cycle = numCycles - 1; cycle >= 0; --cycle) {
    mpicout << "----------- Cycle " << cycle << " ----------" << std::endl;
    const auto &amrHier = amrHierCycles[cycle];
    UnitTimer::getInstance().reset();
    std::unique_ptr<Wrapper_Silo<2>> silo;
    if (enableOutput) {
      silo = std::make_unique<Wrapper_Silo<2>>(
          "output", fileName, amrHier, cycle);
      solCycles[cycle] =
          solveAMRPoisson(solver, amrHier, cycle, silo.get(), middleOutput);
    } else {
      solCycles[cycle] = solveAMRPoisson(solver, amrHier, cycle);
    }

    AMRIntergridOp<2> amrOp(amrHier);
    if (cycle < numCycles - 1) {
      MGIntergrid<2> mgOp;
      auto err = amrHier.createAMRData<Real>(CellCenter, 1);
      for (int lv = 0; lv < numHier; ++lv) {
        mgOp.applyRestrict(solCycles[cycle + 1][lv], err[lv]);
        err[lv] = err[lv] - solCycles[cycle][lv];
      }
      amrOp.averageToCoarse(err);

      Real norm[3];
      for (int i = 0; i <= 2; ++i)
        norm[i] = amrOp.computeNorm(err, i)[0];
      mpicout << "max-norm error: " << norm[0] << std::endl;
      mpicout << "1-norm error: " << norm[1] << std::endl;
      mpicout << "2-norm error: " << norm[2] << std::endl;

      if (enableOutput)
        silo->putAMRScalar(err, "error");
    }

    if (enableOutput) {
      auto d2phi = amrHier.createAMRData<Real>(CellCenter, 2);
      auto gradcc = amrHier.createAMRData<Real>(CellCenter, 2);
      auto gradmag = amrHier.createAMRData<Real>(CellCenter);
      for (int lv = 0; lv < amrHier.size(); ++lv) {
        auto lvOp = amrHier.createLevelOp(lv);
        lvOp.computeD2(solCycles[cycle][lv], d2phi[lv]);
        lvOp.computeGradient(solCycles[cycle][lv], gradcc[lv]);
        lvOp.computeMagnitude(gradcc[lv], gradmag[lv]);
      }
      amrOp.averageToCoarse(gradcc);
      amrOp.averageToCoarse(gradmag);

      silo->putAMRScalar(gradcc, "grad_x", 0);
      silo->putAMRScalar(gradcc, "grad_y", 1);
      silo->putAMRScalar(gradmag, "grad_magnitude");
      silo->putAMRScalar(d2phi, "d2_x", 0);
      silo->putAMRScalar(d2phi, "d2_y", 1);
      auto levelTag = amrHier.createAMRData<int>(CellCenter, 1);
      for (int lv = 0; lv < amrHier.size(); ++lv)
        levelTag[lv] = lv;
      silo->putAMRScalar(levelTag, "AMRLevelTag");
      silo->close();
      silo.release();
    }
    if (timing)
      UnitTimer::getInstance().report();
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  Main(argc, argv);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}