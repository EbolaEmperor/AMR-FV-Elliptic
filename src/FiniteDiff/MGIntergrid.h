#pragma once

#include "AMRTools/LevelData.h"

template <int Dim>
class MGIntergrid {
public:
  using LD = LevelData<Real, Dim>;
  using BLD = LevelData<bool, Dim>;

public:
  MGIntergrid() {}

public:
  /**
   * @brief Average the fine MG-datas to the corresponding coarse boxes.
   *
   * @param fineData: the LevelData in the fine implicit MG-mesh
   * @param coarseData: the LevelData in the coarse implicit MG-mesh
   *
   * @note The ref_ratio between adjacent implicit MG-meshes must be 2.
   */
  template <typename T>
  void applyRestrict(const LevelData<T, Dim> &fineData,
                     LevelData<T, Dim> &coarseData) const;

  /**
   * @brief Piecewise constantly interpolate the coarse MG-datas
   *        to the corresponding fine boxes.
   *
   * @param coarseData: the LevelData in the coarse implicit MG-mesh
   * @param fineData: the LevelData in the fine implicit MG-mesh
   *
   * @note The ref_ratio between adjacent implicit MG-meshes must be 2.
   */
  template <typename T>
  void applyInterpolation(const LevelData<T, Dim> &coarseData,
                          LevelData<T, Dim> &fineData) const;

protected:
  /**
   * @brief Check if the inputs are the LevelDatas of adjacent
   *        implicit MG-meshes.
   *        We should check size, nComps, centering and mesh.
   *        Only do checks in DEBUG mode.
   *
   * @param coarseData: the LevelData in the coarse implicit MG-mesh
   * @param fineData: the LevelData in the fine implicit MG-mesh
   */
  template <typename T>
  void checkValidation(const LevelData<T, Dim> &coarseData,
                       const LevelData<T, Dim> &fineData) const;
};