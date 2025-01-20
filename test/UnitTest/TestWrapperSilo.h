// TestWrapperSilo.h
#ifndef TESTWRAPPERSILO_H
#define TESTWRAPPERSILO_H

#include "AMRTools/LevelData.h"
#include "AMRTools/LevelDataExpr.h"
#include "AMRTools/Wrapper_Silo.h"
#include "Core/Wrapper_OpenMP.h"
#include "FiniteDiff/AMRIntergridOp.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "SpatialOp/AMRMultigrid.h"
#include "catch_amalgamated.hpp"

#include <vector>
using namespace std;
typedef Vec<int, 2> iVec;
const int logF = 9;
const int F = 1 << logF;

AMRMeshHierachy<2> createMesh() {
  vector<Box<2>> boxes;
  boxes.emplace_back((iVec){3, 3}, (iVec){7, 7});
  boxes.emplace_back((iVec){3, 8}, (iVec){7, 12});
  boxes.emplace_back((iVec){8, 3}, (iVec){12, 7});

  vector<int> procs = {0, 1, 2};

  DisjointBoxLayout<2> dbl(boxes, procs);
  dbl = dbl.getRefined(F);

  boxes.clear();
  boxes.emplace_back((iVec){0, 0}, (iVec){3, 3});
  boxes.emplace_back((iVec){0, 4}, (iVec){3, 7});
  boxes.emplace_back((iVec){4, 0}, (iVec){7, 3});

  vector<Box<2>> baseDomainBox = {Box<2>((iVec){0, 0}, (iVec){7, 7})};
  ProblemDomain<2> baseDomain(std::move(baseDomainBox), 1. / 8);
  DisjointBoxLayout<2> baseDbl(boxes, procs);
  baseDomain = baseDomain.getRefined(F);
  baseDbl = baseDbl.getRefined(F);

  AMRMeshHierachy<2> amrHier(baseDomain, baseDbl);
  amrHier.push_back(dbl, 2);
  return amrHier;
}

template <typename T>
void fillData(vector<LevelData<T, 2>> &data, T value) {
  // ld 是一个 LevelData
  for (int i = 0; i < data.size(); i++) {
    auto ld = data[i];
    int nComps = ld.getnComps();
    for (auto it = ld.begin(); it.ok(); ++it) {
      for (int comp = 0; comp < nComps; ++comp) {
        auto box = it.getValidBox(comp);
        auto &aData = it.getData()[comp];
        loop_box_2(box, i, j) { aData(i, j) = value; }
      }
    }
  }
}

TEST_CASE("WrapperSilo single component data tests", "[WrapperSilo]") {
  // 创建 AMR 网格层次结构
  if (numProcs() != 3)
    return;
  AMRMeshHierachy<2> amrHier = createMesh();
  SECTION("test for int") {
    vector<LevelData<int, 2>> aDatas;
    for (unsigned i = 0; i < amrHier.size(); i++) {
      aDatas.push_back(amrHier.createLevelData<int>(i, CellCenter));
    }
    for (unsigned i = 0; i < amrHier.size(); i++) {
      aDatas[i].memset(0);
    }
    fillData(aDatas, 5);
    Wrapper_Silo<2> silo(".", "sol", amrHier);

    // 输出数据到文件
    silo.putAMRScalar(aDatas, "aDatas");

    // 关闭输出并保存
    silo.close();

    // 测试通过的判定条件
    REQUIRE(static_cast<bool>(silo) == true);
  }
  SECTION("test for double") {
    vector<LevelData<double, 2>> aDatas;
    for (unsigned i = 0; i < amrHier.size(); i++) {
      aDatas.push_back(amrHier.createLevelData<double>(i, CellCenter));
    }
    for (unsigned i = 0; i < amrHier.size(); i++) {
      aDatas[i].memset(0);
    }
    fillData(aDatas, 10.0);
    Wrapper_Silo<2> silo("output/", "sol", amrHier);

    // 输出数据到文件
    silo.putAMRScalar(aDatas, "aDatas");

    // 关闭输出并保存
    silo.close();

    // 测试通过的判定条件
    REQUIRE(static_cast<bool>(silo) == true);
  }
  SECTION("test for bool") {
    vector<LevelData<bool, 2>> aDatas;
    for (unsigned i = 0; i < amrHier.size(); i++) {
      aDatas.push_back(amrHier.createLevelData<bool>(i, CellCenter));
    }
    for (unsigned i = 0; i < amrHier.size(); i++) {
      aDatas[i].memset(false);
    }
    fillData(aDatas, true);
    Wrapper_Silo<2> silo(".", "sol", amrHier);

    // 输出数据到文件
    silo.putAMRScalar(aDatas, "aDatas");

    // 关闭输出并保存
    silo.close();

    // 测试通过的判定条件
    REQUIRE(static_cast<bool>(silo) == true);
  }
}
#endif  // TESTWRAPPERSILO_H
