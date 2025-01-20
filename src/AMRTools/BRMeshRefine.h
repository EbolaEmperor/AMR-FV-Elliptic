/**
 * @file BRMeshRefine.h
 * @author DingKexin
 * @brief Inheritance of class AMRMeshRefine to five a mesh refine method with
 * BR Algorithm
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#include "AMRTools/AMRMeshRefine.h"
// Constants for Berger-Rigoutsos algorithm

template <int Dim>
class BRMeshRefine : public AMRMeshRefine<Dim> {
private:
  /// @brief the minimum acceptable ratio of the fill cells in a box in the
  /// Berger-Rigoutsos algorithm in \func
  double FillRatio = 0.75;

public:
  using iVec = Vec<int, Dim>;
  template <typename U>
  using Vector = std::vector<U>;

  /**
   * @brief give the vector of boxes we need to refine according to the tags
   * with BR Method
   *
   * @param tags the tag of the grid should be refine, which is given by user
   * @param refBoxes the boxes we need to refine, which is get from the
   * BR-Algorithm
   * @param refRatio
   */

  void makeBoxesParallel(const LevelData<bool, Dim> &tags,
                         Vector<Box<Dim>> &ref_boxes) const;

  /**
   * @brief set the fill ratio of every box, which is defalt set to 0.75
   * @param a_FillRatio after the BR Algotithm each box has:
   * tag cells/all cells > FillRatio
   */
  void setFillRatio(double a_FillRatio) { FillRatio = a_FillRatio; }

public:
  BRMeshRefine(const DisjointBoxLayout<Dim> &mesh,
               Communicator comm = MPI_COMM_WORLD) :
      AMRMeshRefine<Dim>(mesh, comm) {}

private:
  void makeBoxes(Box<Dim> box,
                 const Tensor<bool, Dim> &tags,
                 Vector<Box<Dim>> &refBoxes) const;
  // void splitTagsInBestDimension(LevelData<bool, Dim> &a_tags_inout_lo,
  //                               LevelData<bool, Dim> &a_tags_hi) const;
  bool splitTagsInBestDimension(const Tensor<bool, Dim> &tags,
                                // const Box<Dim> &box,
                                Box<Dim> &box_lo,
                                Box<Dim> &box_hi) const;
  // const std::vector<int> makeTrace(const LevelData<bool, Dim> &a_Ivs,
  //                                  int a_dir) const;
  // void makeTraces(const LevelData<bool, Dim> &a_Ivs,
  //                 std::vector<int> *traces) const;

  int findSplit(const std::vector<int> &a_trace) const;

  int findMxInflectionPoint(const std::vector<int> &a_trace,
                            int &a_maxVal) const;

  // void splitTags(const LevelData<bool, Dim> &a_tags,
  //                const int a_split_dir,
  //                const int a_split_indx,
  //                LevelData<bool, Dim> &a_tags_lo,
  //                LevelData<bool, Dim> &a_tags_hi) const;

  void splitTagsInPlace(const int a_split_dir,
                        const int a_split_indx,
                        Box<Dim> &a_box_lo,
                        Box<Dim> &a_box_hi) const;
  int maxloc(const int *a_V, const int a_size) const;
  int longsideRefineDirs(const Box<Dim> &a_bx, int &a_dir) const;
};
