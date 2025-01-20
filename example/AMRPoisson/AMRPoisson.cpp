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
#include "FunctionFactory.h"
#include "SpatialOp/AMRMultigrid.h"

// Compute and report the error.
// realDatas will be the error after the function call.
void reportError(const AMRMeshHierachy<2> &amrHier,
                 std::vector<LevelData<Real, 2>> &realDatas,
                 const std::vector<LevelData<Real, 2>> &aDatas) {
  // Compute the error with the exact solution.
  // Take average of fine datas to coarse levels
  // to correct values in the coarse mesh covered by fine meshes.
  AMRIntergridOp<2> amrOp(amrHier);
  for (unsigned i = 0; i < amrHier.size(); i++)
    realDatas[i] = realDatas[i] - aDatas[i];
  amrOp.averageToCoarse(realDatas);

  // Compute Linf, L1, L2 norms of errors.
  std::vector<Real> norm[3];
  for (int i = 0; i <= 2; ++i)
    norm[i] = amrOp.computeNorm(realDatas, i);

  int nComps = norm[0].size();
  for (int comp = 0; comp < nComps; ++comp) {
    if (nComps > 1)
      mpicout << "component " << comp << std::endl;
    mpicout << "max-norm error: " << norm[0][comp] << std::endl;
    mpicout << "1-norm error: " << norm[1][comp] << std::endl;
    mpicout << "2-norm error: " << norm[2][comp] << std::endl;
  }
}

std::vector<LevelData<Real, 2>> solveAMRPoisson(
    const lightJSON::jsonNode &solverJS,
    const AMRMeshHierachy<2> &amrHier,
    Wrapper_Silo<2> *silo = nullptr,
    bool middleOutput = false) {
  // Get test functions from the factory.
  FunctionFactory<2> funFac;
  auto pPhi = funFac.getFunc(solverJS["phi"]);
  auto pPhiGrad = funFac.getFunc(solverJS["phiGrad"]);
  auto pPhiLap = funFac.getFunc(solverJS["phiLap"]);

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
  // Set boundary types. bcType[i][D] means the boundary type
  // of the D-th face of the i-th domain box.
  // D for Dirichlet, N for Neumann, I for inner-face.
  std::vector<std::array<char, 4>> bcType(baseDomain.size());
  for (unsigned i = 0; i < bcType.size(); ++i)
    for (int D = 0; D < 4; ++D)
      bcType[i][D] = boundaryCondition[i][D];

  // Initialize FuncFiller of each level.
  std::vector<FuncFiller<2>> func_filler;
  for (unsigned i = 0; i < amrHier.size(); i++)
    func_filler.push_back(amrHier.createFuncFiller(i));

  // Create AMR datas with cell-centered and 1 component.
  // They are vectors.
  // e.g. aDatas[i] is the LevelData of the i-th level.
  auto aDatas = amrHier.createAMRData<Real>(centering);
  auto rhsDatas = amrHier.createAMRData<Real>(centering);
  auto realDatas = amrHier.createAMRData<Real>(centering);

  // Create AMR boundary data, which is a vector.
  // allcDatas[i] is the boundary datas in the i-th level.
  auto allcDatas = amrHier.createAMRBdryData();

  // rhsFunc = Helmoltz(phi) = a * phi + b * Laplace(phi)
  auto rhsFunc = [&](const Vec<Real, 2> &x) {
    Real b = HelmholtzCoef[1] * (*pPhiLap)(x);
    if (HelmholtzCoef[0] != 0)
      b += HelmholtzCoef[0] * (*pPhi)(x);
    return b;
  };

  for (unsigned i = 0; i < amrHier.size(); i++) {
    aDatas[i].memset(0);
    // Fill the rhs datas and exact solution datas
    // with the give functions, using numerical integration method.
    func_filler[i].fillAvr(rhsDatas[i], rhsFunc);
    func_filler[i].fillAvr(realDatas[i], *pPhi);

    // Fill boundary datas in each face of each domain box.
    for (unsigned dm = 0; dm < baseDomain.size(); ++dm)
      for (int faceid = 0; faceid < 4; ++faceid) {
        if (bcType[dm][faceid] == 'D') {
          // For Dirichlet boundary, fill with phi.
          func_filler[i].fillBdryAvr(
              allcDatas[i], *pPhi, dm, faceid, faceid >> 1);
        } else if (bcType[dm][faceid] == 'N') {
          // For Neumann boundaty, fill with normal * Grad(phi).
          Vec<Real, 2> normal = 0.;
          normal[faceid >> 1] = (faceid & 1) ? 1 : -1;
          auto normalGrad = [&](const Vec<Real, 2> &x) {
            return pPhiGrad->dot(x, normal);
          };
          func_filler[i].fillBdryAvr(
              allcDatas[i], normalGrad, dm, faceid, faceid >> 1);
        }
      }
  }

  // Initialize multigrid solver and call the solve function.
  AMRMultigrid<2> multigrid(amrHier);
  multigrid.define(HelmholtzCoef[0],
                   HelmholtzCoef[1],
                   centering,
                   numBottomLevel,
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

  // Report errors.
  mpicout << "Solution ----------------" << std::endl;
  reportError(amrHier, realDatas, aDatas);

  if (silo != nullptr) {
    silo->putAMRScalar(aDatas, "solution");
    silo->putAMRScalar(realDatas, "error");
  }
  return aDatas;
}

void doTestGradient(const std::vector<LevelData<Real, 2>> &solution,
                    const lightJSON::jsonNode &solverJS,
                    const AMRMeshHierachy<2> &amrHier) {
  // Get the vector-valued gradient function from the factory.
  FunctionFactory<2> funFac;
  auto pPhiGrad = funFac.getFunc(solverJS["phiGrad"]);

  // Initialize FuncFiller
  std::vector<FuncFiller<2>> func_filler;
  for (unsigned i = 0; i < amrHier.size(); i++)
    func_filler.push_back(amrHier.createFuncFiller(i));

  // Here we fill exact gradient in control faces,
  // and compare the error with the face-centered numerical gradient.
  auto realGrad = amrHier.createAMRDataFaceVector<Real>();
  for (unsigned i = 0; i < amrHier.size(); i++)
    for (int comp = 0; comp < 2; ++comp) {
      // Extract the comp-th component of the vector-falued gradient function.
      auto compFunc = [&](const Vec<Real, 2> &x) {
        return (*pPhiGrad)(x, comp);
      };
      // Fill the comp-th compnent of the exact gradient.
      func_filler[i].fillAvr(realGrad[i], compFunc, true, comp);
    }

  // Compute the face-centered gradient.
  auto gradfc = amrHier.createAMRDataFaceVector<Real>();
  for (int lv = 0; lv < amrHier.size(); ++lv) {
    auto lvOp = amrHier.createLevelOp(lv);
    lvOp.computeGradient(solution[lv], gradfc[lv]);
  }

  // Report errors.
  mpicout << "Gradient ----------------" << std::endl;
  reportError(amrHier, realGrad, gradfc);
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

  bool testGradient;
  PGET(solver, testGradient);

  // Get the domain and mesh of the coarsest level.
  // And initialize an AMRMeshHierachy with only 1 level.
  auto dwr = DomainFactory<2>::getDomainWrapper(grid);
  AMRMeshHierachy<2> amrHier(dwr->getDomain(), dwr->getMesh());

  for (int cycle = 0; cycle < numHier; ++cycle) {
    UnitTimer::getInstance().reset();
    mpicout << "----------- Cycle " << cycle << " ----------" << std::endl;

    std::vector<LevelData<Real, 2>> solution;
    std::unique_ptr<Wrapper_Silo<2>> silo;
    if (enableOutput) {
      // Initialize the Silo outputer.
      // Output to the file : output/{fileName}.{cycle}.proc{procID}.silo
      silo = std::make_unique<Wrapper_Silo<2>>(
          "output", fileName, amrHier, cycle);
      solution = solveAMRPoisson(solver, amrHier, silo.get(), middleOutput);
    } else {
      solution = solveAMRPoisson(solver, amrHier);
    }

    // Create cell-centered AMR datas.
    // gradcc: cell-centered gradient, 2 components.
    // gradmag: magnitude of cell-centered gradient, 1 component.
    auto gradcc = amrHier.createAMRData<Real>(CellCenter, 2);
    auto gradmag = amrHier.createAMRData<Real>(CellCenter);
    auto tags = amrHier.createAMRData<bool>(CellCenter);

    for (int lv = 0; lv < amrHier.size(); ++lv) {
      auto lvOp = amrHier.createLevelOp(lv);
      // Compute Grad(solution[lv]) and result in grad[lv].
      lvOp.computeGradient(solution[lv], gradcc[lv]);
      // Compute the magnitude of the cell-centered gradient.
      lvOp.computeMagnitude(gradcc[lv], gradmag[lv]);
      // Tag the cell whose gradient magnitude is large than theta*maxgrad
      lvOp.computeTags(gradmag[lv], tags[lv], refThereshold[cycle]);
    }

    // Before outputing, we should take average to coarse levels.
    AMRIntergridOp<2> amrOp(amrHier);
    amrOp.averageToCoarse(gradcc);
    amrOp.averageToCoarse(gradmag);

    // Compute the exact gradient and the numerical error
    if (testGradient)
      doTestGradient(solution, solver, amrHier);

    // Get the finer mesh in the next level.
    DisjointBoxLayout<2> finerMesh;
    if (cycle < numHier - 1) {
      auto tags = amrHier.createAMRData<bool>(CellCenter);
      for (int lv = 0; lv < amrHier.size(); ++lv) {
        BRMeshRefine<2> refiner(amrHier.getMesh(lv));
        // Use the refine tags in the finest level to generate
        // an even finer level mesh.
        if (lv == amrHier.size() - 1) {
          refiner.setFillRatio(fillRatio[cycle]);
          finerMesh = refiner.makeLayout(tags[lv], refRatio[cycle]);
        }
      }
    }

    if (enableOutput) {
      silo->putAMRScalar(gradcc, "grad_x", 0);
      silo->putAMRScalar(gradcc, "grad_y", 1);
      silo->putAMRScalar(gradmag, "grad_magnitude");
      silo->putAMRScalar(tags, "RefineTags");
      // Tag the region in the i-th level mesh by integer i.
      auto levelTag = amrHier.createAMRData<int>(CellCenter, 1);
      for (int lv = 0; lv < amrHier.size(); ++lv)
        levelTag[lv] = lv;
      silo->putAMRScalar(levelTag, "AMRLevelTag");
      silo->close();
      silo.release();
    }
    if (timing)
      UnitTimer::getInstance().report();

    // You should push the finer mesh after Silo output.
    // Because the AMRMeshHierachy in the Wrapper_Silo is a reference.
    // It will cause a Runtime Error if you change the AMRMeshHierachy
    // before deconstructing Wrapper_Silo.
    if (cycle < numHier - 1)
      amrHier.push_back(finerMesh, refRatio[cycle]);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  Main(argc, argv);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}