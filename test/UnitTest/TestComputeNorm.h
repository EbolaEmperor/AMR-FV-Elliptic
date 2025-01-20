#include "catch_amalgamated.hpp"

#include <FiniteDiff/LevelOp.h>

TEST_CASE("2-D ComputeNorm Test with 2 procs", "[LinearSolver]") {
  std::vector<Box<2>> boxes;
  boxes.emplace_back((Vec<int, 2>){0, 0}, (Vec<int, 2>){1, 1});
  boxes.emplace_back((Vec<int, 2>){2, 1}, (Vec<int, 2>){3, 2});
  boxes.emplace_back((Vec<int, 2>){0, 2}, (Vec<int, 2>){1, 3});
  std::vector<int> procs = {0, 1, 0};
  int nProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  if (nProcs == 1)
    return;
  DisjointBoxLayout<2> dbl(std::move(boxes), std::move(procs));

  std::vector<Box<2>> domains;
  domains.emplace_back((Vec<int, 2>){0, 0}, (Vec<int, 2>){4, 4});
  ProblemDomain<2> domain(domains, 0.5, 2);

  LevelOp solver(domain, MPI_COMM_WORLD);

  std::vector<int> centering{-2, -1, 0, 1};
  LevelData<Real, 2> data(dbl, centering, centering.size(), 2, MPI_COMM_WORLD);

  int cnt = 0.0;
  for (auto itr = data.begin(); itr.ok(); ++itr) {
    for (auto &d : itr.getData()) {
      d = ProcID() + cnt;
    }
    cnt += 10.0;
  }

  SECTION("different centering test for q=0") {
    auto norm = solver.computeNorm(data, 0);
    REQUIRE_THAT(norm[1], Catch::Matchers::WithinAbs(10.0, 1e-10));
    REQUIRE_THAT(norm[0], Catch::Matchers::WithinAbs(10.0, 1e-10));
    REQUIRE_THAT(norm[2], Catch::Matchers::WithinAbs(10.0, 1e-10));
    REQUIRE_THAT(norm[3], Catch::Matchers::WithinAbs(10.0, 1e-10));
  }

  SECTION("different centering test for q=1") {
    auto norm = solver.computeNorm(data, 1);
    REQUIRE_THAT(norm[0], Catch::Matchers::WithinAbs(16.5, 1e-10));
    REQUIRE_THAT(norm[1], Catch::Matchers::WithinAbs(11, 1e-10));
    REQUIRE_THAT(norm[2], Catch::Matchers::WithinAbs(16, 1e-10));
    REQUIRE_THAT(norm[3], Catch::Matchers::WithinAbs(11.5, 1e-10));
  }

  SECTION("different centering test for q=2") {
    auto norm = solver.computeNorm(data, 2);
    REQUIRE_THAT(norm[0], Catch::Matchers::WithinAbs(std::sqrt(151.5), 1e-10));
    REQUIRE_THAT(norm[1], Catch::Matchers::WithinAbs(std::sqrt(101.0), 1e-10));
    REQUIRE_THAT(norm[2], Catch::Matchers::WithinAbs(std::sqrt(151), 1e-10));
    REQUIRE_THAT(norm[3], Catch::Matchers::WithinAbs(std::sqrt(101.5), 1e-10));
  }
}