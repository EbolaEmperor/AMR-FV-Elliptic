#include <AMRTools/AMRMeshHierachy.h>
#include <AMRTools/BRMeshRefine.h>
#include <AMRTools/LevelData.h>
#include <AMRTools/Wrapper_Silo.h>
#include <FiniteDiff/AMRIntergridOp.h>
#include <cmath>
#include <vector>
using namespace std;
using iVec = Vec<int, 2>;
using rVec = Vec<double, 2>;

const int logF = 5;
const int F = 1 << logF;

AMRMeshHierachy<2> createMesh() {
  vector<Box<2>> boxes;
  boxes.emplace_back((iVec){0, 0}, (iVec){3, 3});
  boxes.emplace_back((iVec){0, 4}, (iVec){3, 7});
  boxes.emplace_back((iVec){4, 0}, (iVec){7, 3});
  boxes.emplace_back((iVec){4, 4}, (iVec){7, 7});
  vector<int> procs(4, 0);
  DisjointBoxLayout<2> dbl(boxes, procs);
  ProblemDomain<2> pd(boxes, 1. / 8, 2);
  AMRMeshHierachy<2> amrHier(pd.getRefined(F), dbl.getRefined(F));
  return amrHier;
}

LevelData<bool, 2> produceTags(const AMRMeshHierachy<2> &amrHier) {
  const double p = 40;
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
  auto fcfiller = amrHier.createFuncFiller(0);
  auto grad = amrHier.createLevelData<Real>(0, CellCenter);
  fcfiller.fillAvr(grad, f);
  BRMeshRefine<2> refiner(amrHier.getMesh(0));
  auto tag = amrHier.createLevelData<bool>(0, CellCenter);
  auto lvOp = amrHier.createLevelOp(0);
  lvOp.computeTags(grad, tag, 0.6);
  return tag;
}

void testBR() {
  auto amrHier = createMesh();
  auto tags = produceTags(amrHier);
  BRMeshRefine<2> refiner(amrHier.getMesh(0));
  refiner.setFillRatio(0.5);
  vector<Box<2>> refs;
  amrHier.push_back(refiner.makeLayout(tags, 4), 4);
  Wrapper_Silo<2> silo(".", "test", amrHier);

  vector<LevelData<bool, 2>> ltgs;
  for (unsigned i = 0; i < amrHier.size(); ++i)
    ltgs.push_back(amrHier.createLevelData<bool>(i, CellCenter));
  ltgs[0] = refiner.enlargeTags(tags);
  AMRIntergridOp amrOp(amrHier);
  ltgs[1].memset(0);
  amrOp.constantInterpolateIncr(ltgs[0], ltgs[1], 1);
  silo.putAMRScalar(ltgs, "tags");
}

int main() {
  MPI_Init(NULL, NULL);
  testBR();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
