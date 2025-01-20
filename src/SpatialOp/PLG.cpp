#include "SpatialOp/PLG.h"

#include "Core/TensorSlice.h"
#include "Core/VecCompare.h"
#include "Core/numlib.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <set>

template <int D, int n>
typename PLG<D, n>::TestSet PLG<D, n>::testSet;

template <int D, int n>
void PLG<D, n>::initialize() {
  if (testSet.initialized)
    return;
  dbgcout2 << "Initializing the test sets for D = " << D << ", n = " << n
           << " : ";
  testSet.initialized = true;
  auto &identity = testSet.identity;
  auto &travOrder = testSet.travOrder;
  auto &locations = testSet.locations;
  auto &bookmarks = testSet.bookmarks;
  // 1. First initialize the traverse order.
  static_assert(D == 2 || D == 3,
                "Dimensions other than 2, 3 are not supported. ");
  for (int d = 0; d < D; ++d)
    std::iota(&identity[d][0], &identity[d][n] + 1, 0);
  travOrder = identity;
  // 2. Then initialize the test sets.
  bookmarks.push_back(0);
  std::vector<iVec> principal = form(identity);
  std::set<iVec, VecCompare<int, D>> remain(principal.cbegin(),
                                            principal.cend());
  iVec maxCoord;
  for (int t = 0; t < D * (n + 1); ++t) {
    int d = t % D, i = t / D;
    std::vector<iVec> newly;
    for (const auto &p : remain) {
      for (int m = 0; m < D; ++m) {
        int detSize = i + 1 - (m > d);
        auto j = std::find(&travOrder[m][0], &travOrder[m][detSize], p[m]);
        if (j == &travOrder[m][detSize])
          maxCoord[m] =
              *std::max_element(&travOrder[m][detSize], &travOrder[m][n] + 1);
        else
          maxCoord[m] = p[m];
      }
      if (sum(maxCoord) <= n)
        newly.push_back(p);
    }
    dbgcout2 << newly.size() << " ";
    for (const auto &p : newly)
      remain.erase(p);
    locations.insert(locations.cend(), newly.cbegin(), newly.cend());
    bookmarks.push_back(locations.size());
  }
  dbgcout2 << std::endl;
}

template <int D, int n>
auto PLG<D, n>::generate(const Tensor<int, D> &K,
                         const iVec &q,
                         const rVec &anchor,
                         const DnPerm &testOrd,
                         Real &minCpt,
                         int maxNumSol) -> DnPerm {
  assert(testSet.initialized);
  assert(Box<D>(0, n).contain(q));
  assert(K(q));

  // some declarations
  DnPerm A = testSet.identity;
  unsigned int isQChecked = 0;
  VecCompare<int, D> vcmp;
  DnPerm sol;
  minCpt = std::numeric_limits<Real>::max();

  // back-track algorithm
  for (int t = 0;;) {
    // check if the current node is legal
    isQChecked &= ~1;
    bool reject = false;
    if (!reject) {
      for (int i = testSet.bookmarks[t]; i < testSet.bookmarks[t + 1]; ++i) {
        auto p = testSet.locations[i];
        for (int d = 0; d < D; ++d)
          p[d] = testOrd[d][A[d][p[d]]];
        if (K(p) <= 0) {
          reject = true;
          break;
        }
        isQChecked |= (vcmp.compare(p, q) == 0);
      }
    }
    if (!reject) {
      // Case 1 : move on to the next level and initialize.
      if (t + 1 < D * n) {
        ++t;
        initnext(&A[t % D][0], t / D);
        isQChecked <<= 1;
        continue;
      }
      // Case 2 : a solution is found.
      if (t + 1 == D * n && isQChecked) {
        DnPerm temp = inversepermute(testOrd, A);
        Real c = calculateCompactness(temp, anchor);
        if (c < minCpt) {
          minCpt = c;
          sol = temp;
          if (--maxNumSol == 0)
            break;
        }
      }
    }
    // Case 3 : move to the sibling or move up.
    while (t >= 0) {
      if (next(&A[t % D][0], t / D))
        break;
      --t;
      isQChecked >>= 1;
    }
    // Case 4 : all the possibilities have been exhausted, so exit.
    if (t < 0)
      break;
  }
  dbgcout2 << "Optimal compactness = " << minCpt << "\n";
  if (minCpt == std::numeric_limits<Real>::max())
    minCpt = -1.0;
  return sol;
}

template <int D, int n>
void PLG<D, n>::initnext(int *perm, int pos) {
  // Force perm[pos] to be the smallest among [pos...n]
  auto j = std::min_element(&perm[pos], &perm[n + 1]);
  std::iter_swap(&perm[pos], j);
}

template <int D, int n>
bool PLG<D, n>::next(int *perm, int pos) {
  // Find the immediate successor of perm[pos] among [pos+1...n]
  int a = n + 1, j;
  for (int k = pos + 1; k <= n; ++k) {
    if (perm[k] > perm[pos] && perm[k] < a) {
      j = k;
      a = perm[k];
    }
  }
  if (a != n + 1) {  // Upon found
    std::swap(perm[pos], perm[j]);
    return true;
  }
  return false;  // Not found
}

template <int D, int n>
auto PLG<D, n>::permute(const DnPerm &A, const DnPerm &P) -> DnPerm {
  // Permute rows of A by P
  DnPerm B;
  for (int d = 0; d < D; ++d)
    for (int i = 0; i <= n; ++i)
      B[d][P[d][i]] = A[d][i];
  return B;
}

template <int D, int n>
auto PLG<D, n>::inversepermute(const DnPerm &A, const DnPerm &P) -> DnPerm {
  // Permute rows of A by P^{-1}
  DnPerm B;
  for (int d = 0; d < D; ++d)
    for (int i = 0; i <= n; ++i)
      B[d][i] = A[d][P[d][i]];
  return B;
}

template <int D, int n>
Real PLG<D, n>::calculateCompactness(const DnPerm &A, const rVec &q) {
  Real sum = 0;
  for (int d = 0; d < D; ++d)
    for (int i = 0; i <= n; ++i)
      sum +=
          (n + 1 - i) * (ipow<2>(A[d][i] - q[d]) + std::abs(A[d][i] - q[d]));
  return sum;
}

template <int D, int n>
auto PLG<D, n>::generateTestOrder(const Tensor<int, D> &K,
                                  const rVec &q,
                                  TestOrder t,
                                  bool &earlyReject) -> DnPerm {
  assert(testSet.initialized);
  earlyReject = false;
  DnPerm myOrder;
  struct sliceinfo {
    int idx;
    int num;
    Real dist;
  };
  for (int d = 0; d < D; ++d) {
    sliceinfo si[n + 1];
    for (int i = 0; i <= n; ++i) {
      si[i].idx = i;
      si[i].num = sum(K.slice(d, i));
      si[i].dist = std::abs(i - q[d]);
    }
    // First sort by feasibility to determine earlyReject
    std::sort(
        si, si + (n + 1), [](const sliceinfo &lhs, const sliceinfo &rhs) {
          return lhs.num > rhs.num ||
                 (lhs.num == rhs.num && lhs.dist < rhs.dist);
        });
    for (int i = 0; i <= n; ++i)
      earlyReject |= si[i].num < (n + 1 - i);
    // Then sort by compactness on request
    if (t == TestOrder::oneNormFirst)
      std::sort(
          si, si + (n + 1), [](const sliceinfo &lhs, const sliceinfo &rhs) {
            return lhs.dist < rhs.dist ||
                   (lhs.dist == rhs.dist && lhs.num > rhs.num);
          });
    for (int i = 0; i <= n; ++i)
      myOrder[d][i] = si[i].idx;
  }
  return myOrder;
}

template <int D, int n>
auto PLG<D, n>::form(const DnPerm &A) -> std::vector<iVec> {
  // Form the lattice nodes corresponding to the permutation A.
  std::vector<iVec> r;
  iVec u;
  auto pushw = [&r, &A](const iVec &u) {
    iVec w;
    for (int d = 0; d < D; ++d)
      w[d] = A[d][u[d]];
    r.push_back(w);
  };
  for (u[0] = 0; u[0] <= n; ++u[0]) {
    for (u[1] = 0; u[1] <= n - u[0]; ++u[1]) {
      if (D >= 3) {
        for (u[2] = 0; u[2] <= n - (u[0] + u[1]); ++u[2])
          pushw(u);
      } else {
        pushw(u);
      }
    }
  }
  return r;
}

//============================================================
// template class PLG<2,3>;
template class PLG<2, 4>;
// template class PLG<2,5>;
