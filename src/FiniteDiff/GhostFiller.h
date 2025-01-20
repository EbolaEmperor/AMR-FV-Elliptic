#pragma once

#include "AMRTools/DisjointBoxLayout.h"
#include "AMRTools/LevelData.h"
#include "AMRTools/LevelGhostBoxes.h"
#include "AMRTools/ProblemDomain.h"

template <int Dim>
class GhostFiller {
public:
  template <typename T, int D>
  using Array = std::array<T, D>;
  template <typename T>
  using Vector = std::vector<T>;
  using iVec = Vec<int, Dim>;

private:
  ProblemDomain<Dim> domain_;
  DisjointBoxLayout<Dim> mesh_;
  LevelGhostBoxes<Dim> lgb_;

public:
  GhostFiller() {}

  GhostFiller(const ProblemDomain<Dim> &domain,
              const DisjointBoxLayout<Dim> &mesh,
              const LevelGhostBoxes<Dim> &lgb) :
      domain_(domain), mesh_(mesh), lgb_(lgb) {}

public:
  /**
   * @brief Fill all the ghosts with homogeneous condition
   *
   * @param aData: The LevelData to fill, including the centering.
   * @param bcType: discribe the condition type (D or N)
   * @param comp: The component to fill
   */
  void fillGhosts(LevelData<Real, Dim> &aData,
                  Vector<Array<char, Dim * 2>> bcType,
                  unsigned comp = 0) const;

  void fillGhostsOd2(LevelData<Real, Dim> &aData,
                     Vector<Array<char, Dim * 2>> bcType,
                     unsigned comp = 0) const;

  /**
   * @brief Fill all the C-F intergrid ghosts with homogeneous assumption.
   *
   * @param aData: The LevelData to fill, including the centering.
   * @param comp: The component to fill
   * @param uesZeroGhost: If you set to be true, then the C-F ghosts will be
   *                      filled with all zeros. Otherwise, we will suppose
   *                      the face near the far-side ghost is zero,
   *                      and do a 4th order interpolation.
   */
  void fillGhostsCFHomo(LevelData<Real, Dim> &aData,
                        unsigned comp = 0,
                        bool useZeroGhost = false) const;

  /**
   * @brief Fill all the ghosts with the given condition
   *
   * @param aData: The LevelData to fill, including the centering.
   * @param cDatas[d][f](i): The i-th boundary condition value
   *                         of the f-th face of the d-th domain box
   * @param bcType: discribe the condition type (D or N)
   * @param comp: The component to fill
   */
  void fillGhosts(LevelData<Real, Dim> &aData,
                  const Vector<Array<Tensor<Real, Dim - 1>, Dim * 2>> &cDatas,
                  Vector<Array<char, Dim * 2>> bcType,
                  unsigned comp = 0) const;

  /**
   * @brief Fill all the corner ghosts.
   *
   * @param aData: The LevelData to fill, including the centering.
   */
  void fillCorners(LevelData<Real, Dim> &aData) const;

protected:
  /**
   * @brief Fill all the ghosts on some side of a Tensor
   *
   * @param aData: The Tensor to fill.
   * @param D: The direction, 0 for x-direction and 1 for y-direction
   * @param side: -1 for low-side and 1 for high-side
   * @param type: discribe the condition type (D or N)
   * @param cData: The boundary conditions
   * @param centering: The centering of the data
   */
  template <typename T>
  void doFillGhosts(Tensor<Real, Dim> &data,
                    int D,
                    int side,
                    char type,
                    const T &cData,
                    int centering) const;

  template <typename T>
  void doFillGhostsOd2(Tensor<Real, Dim> &data,
                       int D,
                       int side,
                       char type,
                       const T &cData,
                       int centering) const;
};