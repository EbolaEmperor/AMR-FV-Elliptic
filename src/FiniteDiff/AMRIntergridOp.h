#pragma once

#include "AMRTools/AMRMeshHierachy.h"
#include "AMRTools/LevelData.h"
#include "FiniteDiff/GhostFiller.h"
#include "FiniteDiff/LevelOp.h"

#include <array>
#include <vector>

template <int Dim>
class AMRIntergridOp {
public:
  using iVec = Vec<int, Dim>;
  using LD = LevelData<Real, Dim>;
  template <typename T>
  using Vector = std::vector<T>;
  template <typename T, int P>
  using Array = std::array<T, P>;

private:
  const AMRMeshHierachy<Dim> &amrHier_;
  Vector<LevelOp<Dim>> lvOp_;
  Vector<GhostFiller<Dim>> gstFiller_;

public:
  AMRIntergridOp(const AMRMeshHierachy<Dim> &amrHier) : amrHier_(amrHier) {
    lvOp_.resize(amrHier.size());
    gstFiller_.resize(amrHier.size());
    for (unsigned i = 0; i < amrHier.size(); ++i) {
      lvOp_[i] = amrHier_.createLevelOp(i);
      gstFiller_[i] = amrHier_.createGhostFiller(i);
    }
  }

public:
  /**
   * @brief Compute ||aData||_q
   * @return the q-norm of each comonent of aData.
   */
  Vector<Real> computeNorm(const Vector<LD> &aDatas, int q) const;

  /**
   * @brief Exchange the datas in the same level.
   *
   * @param aDatas: the LevelDatas in each level.
   */
  void exchangeInLevel(Vector<LD> &aDatas) const {
    for (auto &aData : aDatas)
      aData.exchangeAll();
  }

  /**
   * @brief Fill all ghosts, including boundary ghosts and intergrid ghosts.
   */
  void fillGhosts(
      Vector<LD> &aDatas,
      const Vector<Vector<Array<Tensor<Real, Dim - 1>, 2 * Dim>>> &cDatas,
      const Vector<Array<char, 2 * Dim>> &bcType) const;

  /**
   * @brief Fill all ghosts, including boundary ghosts and intergrid ghosts,
   *        with homogeneous conditions.
   */
  void fillGhosts(Vector<LD> &aDatas,
                  const Vector<Array<char, 2 * Dim>> &bcType) const;

  /**
   * @brief Average all the fine datas to the coarse box.
   *
   * @param aDatas: the LevelDatas in each level.
   */
  void averageToCoarse(Vector<LD> &aDatas) const {
    for (int i = aDatas.size() - 2; i >= 0; --i) {
      averageToCoarse(aDatas[i + 1], aDatas[i], i + 1);
    }
  }

  /**
   * @brief Average the fine datas to the corresponding coarse boxes.
   *
   * @param fineData: the LevelData in the fine mesh
   * @param coarseData: the LevelData in the coarse mesh
   * @param fineLevel: the fine level index
   */
  void averageToCoarse(const LD &fineData,
                       LD &coarseData,
                       unsigned fineLevel) const;

  /**
   * @brief Interpolate the coarse datas to get corresponding fine ghosts
   *        in the given level.
   *
   * @param coarseData: the LevelData in the coarse mesh
   * @param fineData: the LevelData in the fine mesh
   * @param fineLevel: the fine level index
   *
   * @note Only support cell-center now.
   */
  void interpolateToFineGhost(const LD &coarseData,
                              LD &fineData,
                              unsigned fineLevel) const;

  /**
   * @brief Interpolate the coarse datas to get corresponding fine ghosts
   *        in the given level, with a stencil of order 5.
   *
   * @param coarseData: the LevelData in the coarse mesh
   * @param fineData: the LevelData in the fine mesh
   * @param fineLevel: the fine level index
   *
   * @note No implementation yet.
   */
  void interpolateToFineGhostOd5(const LD &coarseData,
                                 LD &fineData,
                                 unsigned fineLevel) const {};

  /**
   * @brief Piecewise constantly interpolate the coarse datas
   *        and add to corresponding fine datas.
   *        Designed for AMRMultigrid.
   *
   * @param coarseData: the LevelData in the coarse mesh
   * @param fineData: the LevelData in the fine mesh
   * @param fineLevel: the fine level index
   */
  template <typename R>
  void constantInterpolateIncr(const LevelData<R, Dim> &coarseData,
                               LevelData<R, Dim> &fineData,
                               unsigned fineLevel) const;

  /**
   * @brief Piecewise interpolate the coarse datas
   *        and add to corresponding fine datas.
   *        Automatically choose a stencil of the given order.
   *
   * @param coarseData: the LevelData in the coarse mesh
   * @param fineData: the LevelData in the fine mesh
   * @param fineLevel: the fine level index
   * @param order: the interpolation order
   *
   */
  void autoInterpolateIncr(const LD &coarseData,
                           LD &fineData,
                           unsigned fineLevel,
                           int order) const;

private:
  /**
   * @brief Check whether the fine-data and the coarse-data is coincidate,
   *        and whether the meshes are coincidate with amrHier_.
   *
   * @param coarseData: the LevelData in the coarse mesh
   * @param fineData: the LevelData in the fine mesh
   * @param fineLevel: the fine level index
   */
  template <typename R>
  void checkValidation(const LevelData<R, Dim> &coarseData,
                       const LevelData<R, Dim> &fineData,
                       unsigned fineLevel) const;

  /**
   * @brief Fill the corner ghost of aData with a regular O(h^5) stencil
   *
   * @param aData: the LevelData in the level mesh
   * @param level: the level
   */
  void fillCorner(LD &aData, unsigned level) const;

  /**
   * @brief The implementation of linearInterpolationIncr and
   *        quarticInterpolationIncr, or more in the future.
   *
   * @param Order: the order of interpolation polymonial
   * @param nAdd: the number of additional points
   *
   * @note We will use C((Order+2),2)+nAdd points to interpolate
   *       the valid cells, faces or nodes.
   *       NEVER make this function public !!
   *       If you want to add a new interpolation method,
   *       firstly create a cpp file in CFInterp to describe the stencil,
   *       then write a public function to call this function.
   */
  template <int Order, int nAdd>
  void interpolateIncr(const LD &coarseData,
                       LD &fineData,
                       unsigned fineLevel) const;
};