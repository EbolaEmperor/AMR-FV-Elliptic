#include "Core/SimplicialComplex.h"

SimplicialComplex &SimplicialComplex::operator=(const SimplicialComplex &rhs) {
  if (this != &rhs) {
    simplexes.clear();
    mVertex2Simplex.clear();
    for (const auto &sims : rhs.simplexes) {
      for (const auto &s : sims)
        insert(s);
    }
  }
  return *this;
}

int SimplicialComplex::getStarClosure(Vertex p,
                                      SimplicialComplex &closure) const {
  auto sims = mVertex2Simplex.find(p);
  if (sims == mVertex2Simplex.end())
    return 0;
  for (auto sIt : sims->second) {
    closure.insert(*sIt);
  }
  return 1;
}

int SimplicialComplex::getLink(Vertex p,
                               std::unordered_set<Vertex> &res) const {
  auto sims = mVertex2Simplex.find(p);
  if (sims == mVertex2Simplex.end())
    return 0;
  for (auto sIt : sims->second) {
    for (auto vertex : sIt->vertices) {
      res.insert(vertex);
    }
  }
  return 1;
}

int SimplicialComplex::insert(const Simplex &s) {
  Simplex copy(s);
  return insert(copy);
}

int SimplicialComplex::insert(Simplex &s) {
  int sNSim = s.getDimension();
  if (sNSim == -1)
    return 0;
  if (sNSim >= static_cast<int>(simplexes.size()))
    simplexes.resize(sNSim + 1);
  auto pair = simplexes[sNSim].insert(s);
  auto sIt = pair.first;
  auto b = pair.second;
  if (!static_cast<bool>(b))
    return 0;

  auto vIt = s.vertices.begin();
  while (vIt != s.vertices.end()) {
    size_t vertex = *vIt;
    mVertex2Simplex[vertex].insert(sIt);
    s.vertices.erase(vIt);
    insert(s);
    vIt = s.vertices.insert(vertex).first;
    ++vIt;
  }
  return 1;
}

void SimplicialComplex::findShare(
    const std::vector<
        typename std::unordered_map<Vertex, std::set<SimplexIter>>::iterator>
        &sims,
    std::vector<SimplexIter> &shareSim) {
  std::less<SimplexIter> cmp;
  size_t i = 0;
  size_t j = 0;
  size_t n = sims.size();
  if (n == 1) {
    shareSim.insert(
        shareSim.end(), sims[0]->second.begin(), sims[0]->second.end());
    return;
  }
  std::vector<typename std::set<SimplexIter>::iterator> arr;
  for (i = 0; i < n; ++i) {
    arr.push_back(sims[i]->second.begin());
  }
  i = 0;
  SimplexIter now;
  while (true) {
    now = *(arr[i]);
    j = i;
    ++i;
    i %= n;
    while (i != j) {
      while (arr[i] != sims[i]->second.end() && cmp(*(arr[i]), now)) {
        ++arr[i];
      }
      if (arr[i] == sims[i]->second.end() || *(arr[i]) != now)
        break;

      ++i;
      i %= n;
    }
    if (i == j) {
      shareSim.push_back(now);
      ++arr[i];
    }

    if (arr[i] == sims[i]->second.end())
      break;
  }
}

int SimplicialComplex::erase(const Simplex &s) {
  Simplex copy(s);
  return erase(copy);
}

int SimplicialComplex::erase(Simplex &s) {
  int sNSim = s.getDimension();
  auto b = simplexes[sNSim].find(s);
  if (b == simplexes[sNSim].end())
    return 0;

  std::vector<
      typename std::unordered_map<Vertex, std::set<SimplexIter>>::iterator>
      sims;
  std::vector<SimplexIter> shareSim;
  for (auto vertex : s.vertices) {
    sims.push_back(mVertex2Simplex.find(vertex));
  }
  findShare(sims, shareSim);
  eraseExact(shareSim);
  return 1;
}