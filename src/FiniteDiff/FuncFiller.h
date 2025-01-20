#pragma once

#include "AMRTools/LevelData.h"
#include "AMRTools/ProblemDomain.h"
#include "AMRTools/Utilities.h"
#include "Core/Wrapper_OpenMP.h"
#include "Core/numlib.h"

#include <array>
#include <vector>

template <int Dim>
class FuncFiller {
public:
  template <typename T, int D>
  using Array = std::array<T, D>;
  template <typename T>
  using Vector = std::vector<T>;
  using rVec = Vec<Real, Dim>;

private:
  ProblemDomain<Dim> domain_;
  DisjointBoxLayout<Dim> mesh_;

public:
  FuncFiller() {}

  FuncFiller(const ProblemDomain<Dim> &domain,
             const DisjointBoxLayout<Dim> &mesh) :
      domain_(domain), mesh_(mesh) {}

public:
  /**
   * @brief Use the function expression to fill the control volumes or faces.
   *
   * @param aData: The LevelData to fill, including the centering.
   * @param expr: The function expression, supporting operator() (const
   * rVec&)
   * @param comp: The component to fill
   * @param fillGhosts: Whether to fill the ghost cells
   */
  template <typename TFunc>
  void fillAvr(LevelData<Real, Dim> &aData,
               const TFunc &expr,
               bool fillGhosts = true,
               unsigned comp = 0) const;

  /**
   * @brief Use the function expression to fill the boundary conditions.
   *
   * @param cDatas[d][f](i): The i-th boundary condition value
   *        of the f-th face of the d-th domain box
   * @param expr: The function expression, supporting operator()(const rVec&)
   * @param domainBoxID: The boundary of which domain to fill
   * @param D: which face (0123 for down,up,left,right)
   * @param cent: The centering of the boundary data (except CellCenter)
   */
  template <typename TFunc>
  void fillBdryAvr(Vector<Array<Tensor<Real, Dim - 1>, Dim * 2>> &cDatas,
                   const TFunc &expr,
                   int domainBoxID,
                   int D,
                   int cent) const;

  /**
   * @brief Use the function expression to fill all boundary.
   *
   * @param cDatas[d][f](i): The i-th boundary condition value
   *        of the f-th face of the d-th domain box
   * @param expr: The function expression, supporting operator()(const rVec&)
   * @param dataCent: The centering of the LevelData
   */
  template <typename TFunc>
  void fillBdryAvr(Vector<Array<Tensor<Real, Dim - 1>, Dim * 2>> &cDatas,
                   const TFunc &expr,
                   int dataCent) const;
};

template <>
template <typename TFunc>
void FuncFiller<2>::fillAvr(LevelData<Real, 2> &aData,
                            const TFunc &expr,
                            bool fillGhosts,
                            unsigned comp) const {
  UnitTimer::getInstance().begin("fillAvr");
  rVec dx = domain_.getDx();
  rVec x0 = domain_.getX0();
  double area = dx[0] * dx[1];
  int cent = aData.getCentering(comp);

#ifdef USE_OPENMP
  // Use 4 times threads when running fillAvr.
  // But the number of threads should not be greater than 8.
  int old_nThrs = omp_get_max_threads();
  OMPSwitch omps(std::min(old_nThrs << 2, 8));
#endif

  for (auto it = aData.begin(); it.ok(); ++it) {
    auto box = fillGhosts ? it.getGhostedBox(comp) : it.getValidBox(comp);
    auto &data = it.getData()[comp];

    if (cent == CellCenter) {
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i0, i1) {
        rVec iv = {i0 * dx[0] + x0[0], i1 * dx[1] + x0[1]};
        data(i0, i1) = quad<4>(expr, iv, iv + dx) / area;
      }
    } else if (cent == NodeCenter) {
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i0, i1) {
        data(i0, i1) = expr((rVec){i0 * dx[0] + x0[0], i1 * dx[1] + x0[1]});
      }
    } else {  // Face-Center
#pragma omp parallel for default(shared) schedule(static)
      loop_box_2(box, i0, i1) {
        rVec iv = {i0 * dx[0] + x0[0], i1 * dx[1] + x0[1]};
        auto expr_face = [&](const Real &t) {
          rVec jv = iv;
          jv[cent ^ 1] = t;
          return expr(jv);
        };
        data(i0, i1) =
            quad<4>(expr_face, iv[cent ^ 1], iv[cent ^ 1] + dx[cent ^ 1]) /
            dx[cent ^ 1];
      }
    }
  }
  UnitTimer::getInstance().end("fillAvr");
}

template <>
template <typename TFunc>
void FuncFiller<2>::fillBdryAvr(Vector<Array<Tensor<Real, 1>, 4>> &cDatas,
                                const TFunc &expr,
                                int domainBoxID,
                                int D,
                                int cent) const {
  rVec dx = domain_.getDx();
  rVec x0 = domain_.getX0();
  auto domainBox =
      staggerFromCellCenter(domain_.getLayout().getBox(domainBoxID), cent);
  auto &cData = cDatas[domainBoxID][D];

  int nG = domain_.getNumGhosts();
  auto bdryBox =
      (D & 1) ? domainBox.highSideBox(D >> 1) : domainBox.lowSideBox(D >> 1);
  bdryBox = bdryBox.grow(nG, (D >> 1) ^ 1);

  Box<1> cDataBox(bdryBox.lo()[(D >> 1) ^ 1], bdryBox.hi()[(D >> 1) ^ 1]);
  cData.resize(cDataBox);

  if (cent == CellCenter) {
    throw std::runtime_error("Boundary condition cannot be cell-centered!");
  } else if (cent == NodeCenter) {
    int p = bdryBox.lo()[(D >> 1) ^ 1];
    loop_box_2(bdryBox, i0, i1) {
      cData(p++) = expr((rVec){x0[0] + i0 * dx[0], x0[1] + i1 * dx[1]});
    }
  } else {  // Face-Center
    auto expr_face = [&](const Real &t) {
      rVec jv = bdryBox.lo() * dx + x0;
      jv[cent ^ 1] = t;
      return expr(jv);
    };
    int p = bdryBox.lo()[(D >> 1) ^ 1];
    loop_box_2(bdryBox, i0, i1) {
      rVec iv = {i0 * dx[0] + x0[0], i1 * dx[1] + x0[1]};
      cData(p++) =
          quad<4>(expr_face, iv[cent ^ 1], iv[cent ^ 1] + dx[cent ^ 1]) /
          dx[cent ^ 1];
    }
  }
}

template <>
template <typename TFunc>
void FuncFiller<2>::fillBdryAvr(Vector<Array<Tensor<Real, 1>, 4>> &cDatas,
                                const TFunc &expr,
                                int dataCent) const {
  for (unsigned i = 0; i < domain_.size(); i++)
    for (int D = 0; D < 4; D++) {
      if (dataCent == NodeCenter)
        fillBdryAvr(cDatas, expr, i, D, NodeCenter);
      else if (dataCent == CellCenter)
        fillBdryAvr(cDatas, expr, i, D, D >> 1);
      else {  // the data is Face-Center
        if ((D >> 1) == dataCent)
          fillBdryAvr(cDatas, expr, i, D, dataCent);
        else
          fillBdryAvr(cDatas, expr, i, D, NodeCenter);
      }
    }
}