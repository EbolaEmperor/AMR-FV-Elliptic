#include "Core/Box.h"
#include "catch_amalgamated.hpp"

#include <AMRTools/LevelData.h>

template <class T, class Checker>
bool checkLevelData(const LevelData<T, 2> &data,
                    const T &true_value,
                    const Checker &checker) {
  auto &mesh = data.getMesh();
  auto centering = data.getCentering();
  for (auto itr = data.const_begin(); itr.ok(); ++itr) {
    for (unsigned comp = 0; comp != data.getnComps(); ++comp) {
      auto data_box = itr.getData()[comp].box();
      const auto &box_data = itr.getData()[comp];
      for (auto box_itr = mesh.begin(); box_itr.ok(); ++box_itr) {
        auto valid_box = staggerFromCellCenter(*box_itr, centering[comp]);
        auto covered_box = data_box & valid_box;
        loop_box_2(covered_box, i, j) {
          if (!checker(box_data(i, j), true_value)) {
            return false;
          }
        }  /// loop each cell
      }    /// loop each box
    }      /// loop each comps
  }        /// loop each data
  return true;
}

TEST_CASE("2D LevelData Test", "[LevelData]") {
  auto getProcs = [](int maxProc) {
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    nProcs = std::min(maxProc, nProcs);
    std::vector<int> procs(7, 0);
    if (nProcs > 1)
      procs[1] = procs[4] = 1;
    if (nProcs > 2)
      procs[2] = procs[5] = 1;
    return procs;
  };

  std::vector<Box<2>> boxes;
  boxes.emplace_back((Vec<int, 2>){0, 0}, (Vec<int, 2>){3, 3});
  boxes.emplace_back((Vec<int, 2>){0, 4}, (Vec<int, 2>){3, 7});
  boxes.emplace_back((Vec<int, 2>){4, 0}, (Vec<int, 2>){7, 3});
  boxes.emplace_back((Vec<int, 2>){4, 4}, (Vec<int, 2>){5, 5});
  boxes.emplace_back((Vec<int, 2>){4, 6}, (Vec<int, 2>){5, 7});
  boxes.emplace_back((Vec<int, 2>){6, 4}, (Vec<int, 2>){7, 5});
  boxes.emplace_back((Vec<int, 2>){6, 6}, (Vec<int, 2>){7, 7});
  std::vector<int> centering{-2, -1, 0, 1};
  SECTION("One Proc Communication Test for int") {
    auto procs = getProcs(1);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<int, 2> data(dbl, -1, 1, 2, MPI_COMM_WORLD);
    data = 0;
    for (auto itr = data.begin(); itr.ok(); ++itr) {
      itr.getData()[0].slice(itr.getValidBox()) = 1;
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(data, 1, [](int a, int b) { return a == b; }));
  }

  SECTION("One Proc Communication Test for Real") {
    auto procs = getProcs(1);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<Real, 2> data(dbl, -1, 1, 2, MPI_COMM_WORLD);
    data = 0;
    for (auto itr = data.begin(); itr.ok(); ++itr) {
      itr.getData()[0].slice(itr.getValidBox()) = 1.5;
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(
        data, 1.5, [](Real a, Real b) { return fabs(a - b) < 1e-10; }));
  };

  SECTION("Multi Proc Communication Test for int") {
    auto procs = getProcs(3);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<int, 2> data(dbl, -1, 1, 2, MPI_COMM_WORLD);
    data = 0;
    for (auto itr = data.begin(); itr.ok(); ++itr) {
      itr.getData()[0].slice(itr.getValidBox()) = 1;
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(data, 1, [](int a, int b) { return a == b; }));
  }

  SECTION("Mult Proc Communication Test for Real") {
    auto procs = getProcs(3);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<Real, 2> data(dbl, -1, 1, 2, MPI_COMM_WORLD);
    data = 0;
    for (auto itr = data.begin(); itr.ok(); ++itr) {
      itr.getData()[0].slice(itr.getValidBox()) = 1.5;
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(
        data, 1.5, [](Real a, Real b) { return fabs(a - b) < 1e-10; }));
  };

  SECTION("Single Proc Communication Test for int with multi-comp and "
          "multi-centering") {
    auto procs = getProcs(1);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<int, 2> data(dbl, centering, 4, 2, MPI_COMM_WORLD);
    data = 0;
    for (size_t comp = 0; comp != 4; ++comp) {
      for (auto itr = data.begin(); itr.ok(); ++itr) {
        itr.getData()[comp].slice(itr.getValidBox()) = 1;
      }
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(data, 1, [](int a, int b) { return a == b; }));
  }

  SECTION("Single Proc Communication Test for Real with multi-comp and "
          "multi-centering") {
    auto procs = getProcs(1);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<Real, 2> data(dbl, centering, 4, 2, MPI_COMM_WORLD);
    data = 0;
    for (size_t comp = 0; comp != 4; ++comp) {
      for (auto itr = data.begin(); itr.ok(); ++itr) {
        itr.getData()[comp].slice(itr.getValidBox()) = 1.5;
      }
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(
        data, 1.5, [](Real a, Real b) { return fabs(a - b) < 1e-10; }));
  };

  SECTION("Multi Proc Communication Test for int with multi-comp and "
          "multi-centering") {
    auto procs = getProcs(3);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<int, 2> data(dbl, centering, 4, 2, MPI_COMM_WORLD);
    data = 0;
    for (size_t comp = 0; comp != 4; ++comp) {
      for (auto itr = data.begin(); itr.ok(); ++itr) {
        itr.getData()[comp].slice(itr.getValidBox()) = 1;
      }
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(data, 1, [](int a, int b) { return a == b; }));
  }

  SECTION("Multi Proc Communication Test for Real with multi-comp and "
          "multi-centering") {
    auto procs = getProcs(3);
    DisjointBoxLayout<2> dbl(boxes, procs);
    // mpiStopAll(stop);
    LevelData<Real, 2> data(dbl, centering, 4, 2, MPI_COMM_WORLD);
    data = 0;
    for (size_t comp = 0; comp != 4; ++comp) {
      for (auto itr = data.begin(); itr.ok(); ++itr) {
        itr.getData()[comp].slice(itr.getValidBox()) = 1.5;
      }
    }
    data.exchangeAll();
    REQUIRE(checkLevelData(
        data, 1.5, [](Real a, Real b) { return fabs(a - b) < 1e-10; }));
  };
};