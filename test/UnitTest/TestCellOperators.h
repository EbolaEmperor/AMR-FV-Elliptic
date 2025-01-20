#include "AMRTools/AMRMeshHierachy.h"
#include "AMRTools/BRMeshRefine.h"
#include "AMRTools/LevelDataExpr.h"
#include "Core/MPI.h"
#include "Core/Wrapper_OpenMP.h"
#include "DomainFactory.h"
#include "FiniteDiff/AMRIntergridOp.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "FiniteDiff/LevelOp.h"
#include "FunctionFactory.h"
#include "catch_amalgamated.hpp"

TEST_CASE("2D celloperators test", "[celloperators]") {
  std::string inputJsonStr = R"({
        "grid" : {
            "dimension"     : 2,
            "domain"        : "unit square",
            "boxSize0"      : [16, 16]
        }
    })";
  lightJSON::jsonParser *pParser = new lightJSON::jsonParser;
  pParser->parse(inputJsonStr.c_str());
  pParser->finish();
  const auto &root = pParser->getRoot();
  const auto &grid = root["grid"];

  // Get the domain and mesh of the coarsest level.
  // And initialize an AMRMeshHierachy with only 1 level.
  auto dwr = DomainFactory<2>::getDomainWrapper(grid);
  const ProblemDomain<2> &domain = dwr->getDomain();
  const DisjointBoxLayout<2> &mesh = dwr->getMesh();

  int loop = 4;
  std::vector<std::vector<Real>> norm[2];  // for Laplacian and Holmoltz
  for (auto &inner_vector : norm) {
    inner_vector.resize(loop);
  }
  std::vector<std::vector<Real>> gradnorm[2];  // for Gradient
  for (auto &inner_vector : gradnorm) {
    inner_vector.resize(loop);
  }

  for (int l = 0; l < loop; l++) {
    int ratio = pow(2, l + 1);
    auto meshtemp = mesh.getRefined(ratio);
    auto domaintemp = domain.getRefined(ratio);

    LevelData<Real, 2> u(meshtemp, -1, 1, 2, MPI_COMM_WORLD);
    FuncFiller<2> funcfill(domaintemp, meshtemp);
    LevelOp<2> levelop(domaintemp, MPI_COMM_WORLD);

    auto func = [](const Vec<Real, 2> &x) -> Real {
      return sin(M_PI * x[0]) * sin(M_PI * x[1]);
    };
    auto funcGradX = [](const Vec<Real, 2> &x) -> Real {
      return M_PI * cos(M_PI * x[0]) * sin(M_PI * x[1]);
    };
    auto funcGradY = [](const Vec<Real, 2> &x) -> Real {
      return M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]);
    };
    auto funcLAP = [](const Vec<Real, 2> &x) -> Real {
      return -2 * M_PI * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
    };

    funcfill.fillAvr(u, func, true, 0);
    // std::ofstream file("/root/TESTAMR/output/u.txt",std::ios::out);
    // test Operator Laplacian
    LevelData<Real, 2> lap_u(meshtemp, -1, 1, 2, MPI_COMM_WORLD);
    funcfill.fillAvr(lap_u, funcLAP, true, 0);
    LevelData<Real, 2> cal_lap(meshtemp, -1, 1, 2, MPI_COMM_WORLD);
    cal_lap = 0.0;
    levelop.computeLaplacian(u, cal_lap);
    cal_lap = cal_lap - lap_u;
    for (int p = 0; p < 3; p++)
      norm[0][l].push_back(levelop.computeNorm(cal_lap, p)[0]);

    // test Operator Holmolta
    LevelData<Real, 2> helmoltz = u;
    helmoltz = helmoltz + lap_u;
    LevelData<Real, 2> cal_helmoltz(meshtemp, -1, 1, 2, MPI_COMM_WORLD);
    cal_helmoltz = 0.0;
    levelop.computeHelmoltz(u, cal_helmoltz, 1, 1);
    cal_helmoltz = cal_helmoltz - helmoltz;
    for (int p = 0; p < 3; p++)
      norm[1][l].push_back(levelop.computeNorm(cal_helmoltz, p)[0]);

    // test Operator Gradient Cell to Cell
    LevelData<Real, 2> gradient(meshtemp, {-1, -1}, 2, 2, MPI_COMM_WORLD);
    funcfill.fillAvr(gradient, funcGradX, true, 0);
    funcfill.fillAvr(gradient, funcGradY, true, 1);
    LevelData<Real, 2> cal_gradient(meshtemp, {-1, -1}, 2, 2, MPI_COMM_WORLD);
    cal_gradient = 0.0;
    levelop.computeGradient(u, cal_gradient);
    cal_gradient = cal_gradient - gradient;
    for (int p = 0; p < 3; p++) {
      auto xynorm = levelop.computeNorm(cal_gradient, p);
      gradnorm[0][l].push_back(xynorm[0]);
      gradnorm[0][l].push_back(xynorm[1]);
    }

    // test Operator Gradient Cell to Face
    LevelData<Real, 2> gradient2(meshtemp, {0, 1}, 2, 2, MPI_COMM_WORLD);
    funcfill.fillAvr(gradient2, funcGradX, true, 0);
    funcfill.fillAvr(gradient2, funcGradY, true, 1);
    LevelData<Real, 2> cal_gradient2(meshtemp, {0, 1}, 2, 2, MPI_COMM_WORLD);
    cal_gradient2 = 0.0;
    levelop.computeGradient(u, cal_gradient2);
    cal_gradient2 = cal_gradient2 - gradient2;
    for (int p = 0; p < 3; p++) {
      auto xynorm = levelop.computeNorm(cal_gradient2, p);
      gradnorm[1][l].push_back(xynorm[0]);
      gradnorm[1][l].push_back(xynorm[1]);
    }
  }

  // display error ratio
  for (int i = 0; i < 2; i++) {
    if (i == 0)
      std::cout << "*****Laplacian*****" << std::endl;
    else
      std::cout << "*****Holmoltz*****" << std::endl;
    for (int l = 0; l < loop; l++) {
      if (l == 0)
        std::cout << "infty-norm:" << norm[i][l][0] << " L1:" << norm[i][l][1]
                  << " L2:" << norm[i][l][2] << std::endl;
      else {
        std::cout << "infty-norm:" << norm[i][l][0]
                  << " ratio:" << log2(norm[i][l - 1][0] / norm[i][l][0])
                  << " L1:" << norm[i][l][1]
                  << " ratio:" << log2(norm[i][l - 1][1] / norm[i][l][1])
                  << " L2:" << norm[i][l][2]
                  << " ratio:" << log2(norm[i][l - 1][2] / norm[i][l][2])
                  << std::endl;
        REQUIRE((log2(norm[i][l - 1][0] / norm[i][l][0]) >= 3.95 &&
                 log2(norm[i][l - 1][1] / norm[i][l][1]) >= 3.95 &&
                 log2(norm[i][l - 1][2] / norm[i][l][2]) >= 3.95));
      }
    }
  }
  for (int i = 0; i < 2; i++) {
    if (i == 0)
      std::cout << "*****Gradient Cell to Cell*****" << std::endl;
    else
      std::cout << "*****Gradient Cell to Face*****" << std::endl;
    for (int l = 0; l < loop; l++) {
      if (l == 0) {
        std::cout << "X: infty-norm:" << gradnorm[i][l][0]
                  << " L1:" << gradnorm[i][l][2] << " L2:" << gradnorm[i][l][4]
                  << std::endl;
        std::cout << "Y: infty-norm:" << gradnorm[i][l][1]
                  << " L1:" << gradnorm[i][l][3] << " L2:" << gradnorm[i][l][5]
                  << std::endl;
      } else {
        std::cout << "X: infty-norm:" << gradnorm[i][l][0] << " ratio:"
                  << log2(gradnorm[i][l - 1][0] / gradnorm[i][l][0])
                  << " L1:" << gradnorm[i][l][2] << " ratio:"
                  << log2(gradnorm[i][l - 1][2] / gradnorm[i][l][2])
                  << " L2:" << gradnorm[i][l][4] << " ratio:"
                  << log2(gradnorm[i][l - 1][4] / gradnorm[i][l][4])
                  << std::endl;
        std::cout << "Y: infty-norm:" << gradnorm[i][l][1] << " ratio:"
                  << log2(gradnorm[i][l - 1][1] / gradnorm[i][l][1])
                  << " L1:" << gradnorm[i][l][3] << " ratio:"
                  << log2(gradnorm[i][l - 1][3] / gradnorm[i][l][3])
                  << " L2:" << gradnorm[i][l][5] << " ratio:"
                  << log2(gradnorm[i][l - 1][5] / gradnorm[i][l][5])
                  << std::endl;
        REQUIRE((log2(gradnorm[i][l - 1][0] / gradnorm[i][l][0]) >= 3.95 &&
                 log2(gradnorm[i][l - 1][1] / gradnorm[i][l][1]) >= 3.95 &&
                 log2(gradnorm[i][l - 1][2] / gradnorm[i][l][2]) >= 3.95 &&
                 log2(gradnorm[i][l - 1][3] / gradnorm[i][l][3]) >= 3.95 &&
                 log2(gradnorm[i][l - 1][4] / gradnorm[i][l][4]) >= 3.95 &&
                 log2(gradnorm[i][l - 1][5] / gradnorm[i][l][5]) >= 3.95));
      }
    }
  }
}