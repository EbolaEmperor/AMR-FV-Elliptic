#ifndef SIMPLICIALCOMPLEX_TY
#define SIMPLICIALCOMPLEX_TY

#include <algorithm>
#include <cassert>
#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Simplex {
  using Vertex = size_t;
  std::set<Vertex> vertices;

  // constructor
  Simplex() = default;

  template <class Containor>
  explicit Simplex(const Containor &v) : vertices(v.begin(), v.end()) {}

  template <typename InputIterator>
  Simplex(InputIterator first, InputIterator last) : vertices(first, last) {}

  // accessor
  [[nodiscard]] int getDimension() const {
    return static_cast<int>(vertices.size()) - 1;
  }

  // operator
  bool operator<(const Simplex &rhs) const {
    auto lIt = vertices.begin();
    auto rIt = rhs.vertices.begin();
    auto lend = vertices.end();
    auto rend = rhs.vertices.end();
    while (lIt != lend && rIt != rend) {
      if (*lIt != *rIt)
        return *lIt < *rIt;
      ++lIt, ++rIt;
    }
    return rIt != rend;
  }

  bool operator>(const Simplex &rhs) const { return rhs < *this; }

  bool operator==(const Simplex &rhs) const {
    auto lIt = vertices.begin();
    auto rIt = rhs.vertices.begin();
    auto lend = vertices.end();
    auto rend = rhs.vertices.end();
    while (lIt != lend && rIt != rend) {
      if (*lIt != *rIt)
        return false;
      ++lIt, ++rIt;
    }
    return true;
  }

  // visualization
  void print(std::ostream &os) const {
    os << "(";
    for (auto i : vertices) {
      os << i << ",";
    }
    os << "), ";
  }
};

template <>
struct std::less<std::set<Simplex>::iterator> {
  inline bool operator()(const set<Simplex>::iterator &lhs,
                         const set<Simplex>::iterator &rhs) const {
    return *lhs < *rhs;
  }
  // constexpr bool operator()(const unordered_set<Simplex>::iterator& lhs,
  //                           const unordered_set<Simplex>::iterator& rhs)
  //                           const {
  //   return *lhs < *rhs;
  // }
};

template <>
class std::hash<Simplex> {
  static constexpr std::hash<unsigned int> intHash = {};

public:
  inline std::size_t operator()(const Simplex &s) const noexcept {
    size_t res = 0;
    int n = s.getDimension();
    assert(n >= 0 && "Simplex in hash() should be initialed.");
    for (auto i : s.vertices)
      res ^= (intHash(i) << 1);
    return res;
  }
};

//======================================================================

class SimplicialComplex {
public:
  using Vertex = typename Simplex::Vertex;

protected:
  using SimplexIter = typename std::set<Simplex>::iterator;
  std::vector<std::set<Simplex>> simplexes;
  std::unordered_map<Vertex, std::set<SimplexIter>> mVertex2Simplex;

public:
  // constructor
  SimplicialComplex() = default;
  ;

  template <class Containor>
  explicit SimplicialComplex(const Containor &sims);

  template <typename InputIterator>
  SimplicialComplex(InputIterator first, InputIterator last);

  SimplicialComplex(const SimplicialComplex &rhs) { *this = rhs; }
  SimplicialComplex &operator=(const SimplicialComplex &rhs);

  // accessor
  const std::vector<std::set<Simplex>> &getSimplexes() const {
    return simplexes;
  }

  [[nodiscard]] int getDimension() const {
    return static_cast<int>(simplexes.size()) - 1;
  }

  // get a vertex's star
  int getStarClosure(Vertex p, SimplicialComplex &closure) const;

  int getLink(Vertex p, std::unordered_set<Vertex> &res) const;

  // insert a Simplex
  int insert(const Simplex &s);
  int insert(Simplex &s);

  // erase a Simplex
  int erase(const Simplex &s);
  int erase(Simplex &s);

protected:
  // erase a Simplex, do not consider sub simplex. tool for erase
  template <class Containor>
  int eraseExact(const Containor &containor);

  // find all Simplex appear in every element of sims. tool for erase
  static void findShare(
      const std::vector<
          typename std::unordered_map<Vertex, std::set<SimplexIter>>::iterator>
          &sims,
      std::vector<SimplexIter> &shareSim);

public:
  // determine if contain a Simplex.
  bool contain(const Simplex &s) const {
    return simplexes[s.getDimension()].find(s) !=
           simplexes[s.getDimension()].end();
  }

  // visualization
  void print(std::ostream &os) const {
    os << "{";
    size_t i = 0;
    for (i = 0; i < simplexes.size(); ++i) {
      for (const auto &s : simplexes[i]) {
        s.print(os);
        os << " ";
      }
      os << "\n";
    }
    os << "}" << std::endl;
  }
};

template <class Containor>
SimplicialComplex::SimplicialComplex(const Containor &sims) {
  for (auto &s : sims) {
    insert(s);
  }
}

template <typename InputIterator>
SimplicialComplex::SimplicialComplex(InputIterator first, InputIterator last) {
  while (first != last) {
    insert(*first++);
  }
}

template <class Containor>
int SimplicialComplex::eraseExact(const Containor &containor) {
  for (auto it : containor) {
    for (auto vertex : it->vertices) {
      auto mit = mVertex2Simplex[vertex].find(it);
      mVertex2Simplex[vertex].erase(mit);
    }
    simplexes[it->getDimension()].erase(it);
  }
  while (!simplexes.empty() && simplexes.back().empty())
    simplexes.pop_back();
  return 1;
}

#endif  // !SIMPLICIALCOMPLEX_TY
