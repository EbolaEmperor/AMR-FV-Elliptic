#ifndef PLG_H
#define PLG_H

#include "Core/Tensor.h"
#include "Core/Vec.h"

#include <array>
#include <utility>
#include <vector>

/**
 * @brief Encapsulate the implementation of the PLG algorithm.
 * @tparam D The dimension number.
 * @tparam n The highest degree.
 */
template <int D, int n>
class PLG {
public:
  using iVec = Vec<int, D>;
  using rVec = Vec<Real, D>;
  using DnPerm = std::array<std::array<int, n + 1>, D>;

  enum class TestOrder {
    oneNormFirst,    /** The distance to the starting node as the 1st keyword,
                        the number of feasible nodes within the slice as the
                        2nd. */
    feasibilityFirst /** The number of feasible nodes within the slice as
                        the 1st keyword, the distance to the starting node
                        as the 2nd. */
  };

  /**
   * Initialize the test sets.
   * Must be called for every <D,n> instance before calling other functions.
   */
  static void initialize();

  /**
   * @brief Generate a test order for the problem (K,q).
   * See the enumeration TestOrder.
   * @param K The feasible domain of size (n+1)^D. 0 = outside, 1 = inside.
   * @param q The starting point.
   * @param t One of the enumeration TestOrder.
   * @param earlyReject
   * @return The data storing the test order.
   */
  static DnPerm generateTestOrder(const Tensor<int, D> &K,
                                  const rVec &q,
                                  TestOrder t,
                                  bool &earlyReject);

  /**
   * @brief Find the solutions to the PLG problem (K,q).
   * @param K The feasible domain of size (n+1)^D. 0 = outside, 1 = inside.
   * @param q The starting point.
   * @param anchor The center for calculating the compactness.
   * @param testOrd A D by n+1 matrix in row-major order, with each row a
   * permutation of {0,...,n}.
   * @param minCpt The compactness of the solution. -1.0 if no solution is
   * found.
   * @param maxNumSol When this number of solutions have been found, return
   * immediately.
   * @return The solution to the PLG problem.
   */
  static DnPerm generate(const Tensor<int, D> &K,
                         const iVec &q,
                         const rVec &anchor,
                         const DnPerm &testOrd,
                         Real &minCpt,
                         int maxNumSol = 1);

  /**
   * @brief Return the triangular lattice corresponding to the D-permutation
   * A.
   * @param A A D-permutation.
   * @return The coordinates of the triangular lattice.
   */
  static std::vector<iVec> form(const DnPerm &A);

protected:
  // rows of A permuted by P.
  static DnPerm permute(const DnPerm &A, const DnPerm &P);
  // rows of A permuted by P^{-1}
  static DnPerm inversepermute(const DnPerm &A, const DnPerm &P);

  static Real calculateCompactness(const DnPerm &A, const rVec &q);

  struct TestSet {
    std::vector<iVec> locations;
    std::vector<int> bookmarks;
    DnPerm travOrder;
    //    DnPerm            invTravOrder;
    DnPerm identity;
    bool initialized;
  };
  static TestSet testSet;
  static void initnext(int *perm, int pos);
  static bool next(int *perm, int pos);
};

#endif  // PLG_H
