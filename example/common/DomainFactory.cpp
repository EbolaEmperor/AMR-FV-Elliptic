#include "DomainFactory.h"

#include <bit>

namespace NS4_Domains {

class UnitSquare : public DomainWrapper<2> {
public:
  using DomainWrapper<2>::DomainWrapper;
  UnitSquare(const lightJSON::jsonNode &js) { PGET(js, boxSize0); }

  static DomainWrapper<2> *alloc(const lightJSON::jsonNode &js) {
    return new UnitSquare(js);
  }

  ProblemDomain<2> getDomain() const {
    std::vector<Box<2>> boxes;
    boxes.emplace_back((iVec){0, 0}, boxSize0 - 1);
    auto dx = (Vec<Real, 2>){1. / boxSize0[0], 1. / boxSize0[1]};
    return ProblemDomain<2>(std::move(boxes), dx);
  }

  DisjointBoxLayout<2> getMesh() const {
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (std::popcount((unsigned)nProcs) != 1)
      throw std::runtime_error("Unsupported numProcs.");
    std::vector<Box<2>> boxes;
    boxes.emplace_back((iVec){0, 0}, boxSize0 - 1);
    std::vector<int> procs(nProcs);
    for (int i = 0; i < nProcs; ++i)
      procs[i] = i;
    int dir = 0;
    while (nProcs > 1) {
      boxes = getSplittedBoxes(boxes, 2, dir);
      nProcs >>= 1;
      dir ^= 1;
    }
    return DisjointBoxLayout<2>(std::move(boxes), std::move(procs));
  }
};

class Lshape : public DomainWrapper<2> {
public:
  using DomainWrapper<2>::DomainWrapper;
  Lshape(const lightJSON::jsonNode &js) { PGET(js, boxSize0); }

  static DomainWrapper<2> *alloc(const lightJSON::jsonNode &js) {
    return new Lshape(js);
  }

  ProblemDomain<2> getDomain() const {
    std::vector<Box<2>> boxes;
    boxes.emplace_back((iVec){0, 0}, boxSize0 - 1);
    boxes.emplace_back((iVec){0, boxSize0[1]},
                       (iVec){boxSize0[0] - 1, 2 * boxSize0[1] - 1});
    boxes.emplace_back((iVec){boxSize0[0], 0},
                       (iVec){2 * boxSize0[0] - 1, boxSize0[1] - 1});
    auto dx = (Vec<Real, 2>){1. / boxSize0[0], 1. / boxSize0[1]};
    return ProblemDomain<2>(std::move(boxes), dx / 2);
  }

  DisjointBoxLayout<2> getMesh() const {
    int nProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs != 1 &&
        (nProcs % 3 != 0 || std::popcount((unsigned)(nProcs / 3)) != 1))
      throw std::runtime_error("Unsupported numProcs.");
    std::vector<Box<2>> boxes;
    std::vector<int> procs;
    boxes.emplace_back((iVec){0, 0}, boxSize0 - 1);
    boxes.emplace_back((iVec){0, boxSize0[1]},
                       (iVec){boxSize0[0] - 1, 2 * boxSize0[1] - 1});
    boxes.emplace_back((iVec){boxSize0[0], 0},
                       (iVec){2 * boxSize0[0] - 1, boxSize0[1] - 1});
    if (nProcs == 1)
      procs = {0, 0, 0};
    else {
      procs.resize(nProcs);
      for (int i = 0; i < nProcs; ++i)
        procs[i] = i;
      nProcs /= 3;
      int dir = 0;
      while (nProcs > 1) {
        boxes = getSplittedBoxes(boxes, 2, dir);
        nProcs >>= 1;
        dir ^= 1;
      }
    }
    return DisjointBoxLayout<2>(std::move(boxes), std::move(procs));
  }
};

}  // namespace NS4_Domains

template <>
DomainTable<2>::DomainTable() {
#define ADDDOMAIN(name, classname)                                            \
  allocTable.insert(                                                          \
      std::make_pair(std::string(name), NS4_Domains::classname::alloc))

  ADDDOMAIN("unit square", UnitSquare);
  ADDDOMAIN("L-shape", Lshape);

#undef ADDDOMAIN
}

template class DomainTable<2>;
template class DomainFactory<2>;