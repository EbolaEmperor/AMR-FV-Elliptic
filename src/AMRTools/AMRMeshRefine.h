/**
 * @file AMRMeshRefine.h
 * @author {JiatuYan} ({2513630371@qq.com})
 * @brief Base class for AMR mesh refinement
 * @version 0.1
 * @date 2024-04-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "AMRTools/ProblemDomain.h"
#include "LevelData.h"

template <int Dim>
class AMRMeshRefine : public ParallelDataBase<false, false> {
public:
  template <typename U>
  using Vector = std::vector<U>;
  using iVec = Vec<int, Dim>;
  using BaseClass = ParallelDataBase<false, false>;

private:
  DisjointBoxLayout<Dim> mesh_;

  Vector<Box<Dim>> res_boxes_;

  Vector<int> res_procs_;

public:
  AMRMeshRefine() {}

  /**
   * @brief Construct a new AMRMeshRefine object
   *
   * @param domain: the domain
   * @param mesh: layout
   * @param comm: MPI communicator
   */
  AMRMeshRefine(const DisjointBoxLayout<Dim> &mesh,
                Communicator comm = MPI_COMM_WORLD) :
      BaseClass(comm), mesh_(mesh) {}

  /**
   * @brief refine the mesh according to the given tags.
   *
   * @param tags tags that describe how to refine
   * @param ref_ratio refinement ratio
   * @return DisjointBoxLayout<Dim>
   */
  DisjointBoxLayout<Dim> makeLayout(const LevelData<bool, Dim> &tags,
                                    int ref_ratio = 2);

protected:
  /**
   * @brief according to the tags, get the refined boxes of the present
   * processor.
   *
   * @param tags given tags
   * @param ref_boxes the refined boxes of the present processor
   * @param ref_ratio refinement ratio
   */
  virtual void makeBoxesParallel(const LevelData<bool, Dim> &tags,
                                 Vector<Box<Dim>> &ref_boxes) const = 0;

  /**
   * @brief send the refined boxes to the other processors.
   *
   * @param ref_boxes refined boxes to send
   */
  void sendBoxesParallel(const Vector<Box<Dim>> &ref_boxes);

  /**
   * @brief receive the refined boxes from other processors.
   *
   * @param ref_boxes refined boxes received
   * @param procs proc id of the boxes
   */
  void receiveBoxesParallel(Vector<Box<Dim>> &ref_boxes,
                            std::vector<int> &procs);

  /**
   * @brief linearize the data to send.
   *
   */
  void linearIn() override;

  /**
   * @brief decode the data received.
   *
   */
  void linearOut() override;

public:
  /**
   * @brief coarsen tags by ratio 2, then interpolate back.
   */
  LevelData<bool, Dim> enlargeTags(const LevelData<bool, Dim> &tags,
                                   int ratio = 2) const;
};
