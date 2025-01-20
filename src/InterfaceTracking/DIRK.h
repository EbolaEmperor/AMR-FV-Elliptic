#ifndef _DIRK_H_
#define _DIRK_H_

#include "Core/Tensor.h"
#include "Core/TensorExpr.h"
#include "Core/TensorSlice.h"
#include "Core/Wrapper_LAPACKE.h"
#include "RKButcher.h"
#include "TimeIntegrator.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>
/*
template <int Dim>
std::vector<Vec<Real, Dim>> solveNewton(const VectorFunction<Dim> &v, Real tn,
Real coef, const Eigen::SparseMatrix<Real> &Jacobi, const std::vector<Vec<Real,
Dim>> &rhs, const std::vector<Vec<Real, Dim>> &pts, Real tol)
{
    using Point = Vec<Real, Dim>;

    using SpMat = Eigen::SparseMatrix<Real>;

    int num = pts.size() - 1;
    auto res = pts;

    SpMat mat = -coef * Jacobi;
    for (int i = 0; i < Dim * num; i++)
    {
        mat.coeffRef(i, i) += 1;
    }

    //Eigen::SparseLU<SpMat> solver(mat);
    Eigen::BiCGSTAB<SpMat> solver;
    solver.compute(mat);

    //std::cout << mat << std::endl;

    std::vector<Point> vel;
    Eigen::VectorXd err(Dim * num);
    //Tensor<int, 1> ipiv(Dim * num);

    auto normXd = [](const Eigen::VectorXd err, int n) -> Real
    {
        Real res = 0;
        for(int i=0; i<n; i++)
        {
            res += pow(err(i),2);
        }
        return sqrt(res);
    };

    vel = v(res, tn);
    for (int j = 0; j < num; j++)
    {
        err(j) = res[j][0] - rhs[j][0] - coef * vel[j][0];
        err(j + num) = res[j][1] - rhs[j][1] - coef * vel[j][1];
        if (Dim == 3)
            err(j + 2 * num) = res[j][2] - rhs[j][2] - coef * vel[j][2];
    }

    for (int i = 0; i < 10; i++)
    {
        if (normXd(err, Dim * num) < tol)
            break;
        //auto info = LAPACKE_dgesv(LAPACK_COL_MAJOR, 2 * num, 1, mat.data(), 2
* num, ipiv.data(), err.data(), 2 * num); auto x = solver.solve(err); if
(solver.info() != Eigen::Success)
        {
            //std::cout << info << std::endl;
            throw std::runtime_error("solveNewton() - EigenSparseLU");
        }
        for (int j = 0; j < num; j++)
        {
            res[j][0] -= x(j);
            res[j][1] -= x(j + num);
            if (Dim == 3)
            {
                res[j][2] -= x(j + 2 * num);
            }
        }
        res[num][0] -= x(0);
        res[num][1] -= x(num);
        if (Dim == 3)
        {
            res[num][2] -= x(2 * num);
        }
        vel = v(res, tn);
        for (int j = 0; j < num; j++)
        {
            err(j) = res[j][0] - rhs[j][0] - coef * vel[j][0];
            err(j + num) = res[j][1] - rhs[j][1] - coef * vel[j][1];
            if (Dim == 3)
                err(j + 2 * num) = res[j][2] - rhs[j][2] - coef * vel[j][2];
        }
        if (i == 10)
        {
            throw std::runtime_error("solveNewton() out of range");
        }
    }
    return res;
}
*/
/*
template <int Dim, template <int> class VectorRHS>
std::vector<Vec<Real, Dim>> solveNewton(const VectorRHS<Dim> &v, Real tn, Real
coef, const Tensor<Real, 2> &Jacobi, const std::vector<Vec<Real, Dim>> &rhs,
const std::vector<Vec<Real, Dim>> &pts, Real tol)
{
    using Point = Vec<Real, Dim>;

    int num = pts.size() - 1;
    auto res = pts;

    Tensor<Real, 2> mat = -Jacobi * coef;
    for (int i = 0; i < Dim * num; i++)
    {
        mat(i, i) += 1;
    }

    //std::cout << mat << std::endl;

    std::vector<Point> vel;
    Tensor<Real, 1> err(Dim * num);
    Tensor<int, 1> ipiv(Dim * num);

    auto normXd = [](const Tensor<Real, 1> &err, int n) -> Real
    {
        Real res = 0;
        for (int i = 0; i < n; i++)
        {
            res += pow(err(i), 2);
        }
        return sqrt(res);
    };

    vel = v(res, tn);
    for (int j = 0; j < num; j++)
    {
        err(j) = res[j][0] - rhs[j][0] - coef * vel[j][0];
        err(j + num) = res[j][1] - rhs[j][1] - coef * vel[j][1];
        if (Dim == 3)
            err(j + 2 * num) = res[j][2] - rhs[j][2] - coef * vel[j][2];
    }

    for (int i = 0; i < 10; i++)
    {
        if (normXd(err, Dim * num) < tol)
            break;
        auto info = LAPACKE_dgesv(LAPACK_COL_MAJOR, Dim * num, 1, mat.data(),
Dim * num, ipiv.data(), err.data(), Dim * num);

        if (info != 0)
        {
            std::cout << info << std::endl;
            throw std::runtime_error("solveNewton() - LAPACKE_dgesv");
        }
        for (int j = 0; j < num; j++)
        {
            res[j][0] -= err(j);
            res[j][1] -= err(j + num);
            if (Dim == 3)
            {
                res[j][2] -= err(j + 2 * num);
            }
        }
        res[num][0] -= err(0);
        res[num][1] -= err(num);
        if (Dim == 3)
        {
            res[num][2] -= err(2 * num);
        }
        vel = v(res, tn);
        for (int j = 0; j < num; j++)
        {
            err(j) = res[j][0] - rhs[j][0] - coef * vel[j][0];
            err(j + num) = res[j][1] - rhs[j][1] - coef * vel[j][1];
            if (Dim == 3)
                err(j + 2 * num) = res[j][2] - rhs[j][2] - coef * vel[j][2];
        }
        if (i == 10)
        {
            throw std::runtime_error("solveNewton() out of range");
        }
    }
    return res;
}
*/

// VectorFunction version

template <int Dim>
Vec<Real, Dim> solveNewton(const VectorFunction<Dim> &v,
                           Real tn,
                           Real coef,
                           const Tensor<Real, 2> &Jacobi,
                           const Vec<Real, Dim> &rhs,
                           const Vec<Real, Dim> &pt,
                           Real tol) {
  using Point = Vec<Real, Dim>;
  auto res = pt;

  Tensor<Real, 2> mat = -Jacobi * coef;
  for (int i = 0; i < Dim; i++) {
    mat(i, i) += 1;
  }

  Point vel;
  Tensor<Real, 1> err(Dim);
  Tensor<int, 1> ipiv(Dim);

  auto normXd = [](const Tensor<Real, 1> &err, int n) -> Real {
    Real res = 0;
    for (int i = 0; i < n; i++) {
      res += pow(err(i), 2);
    }
    return sqrt(res);
  };

  vel = v(res, tn);
  err(0) = res[0] - rhs[0] - coef * vel[0];
  err(1) = res[1] - rhs[1] - coef * vel[1];
  if (Dim == 3)
    err(2) = res[2] - rhs[2] - coef * vel[2];

  for (int i = 0; i < 10; i++) {
    if (normXd(err, Dim) < tol)
      break;
    auto info = LAPACKE_dgesv(LAPACK_COL_MAJOR,
                              Dim,
                              1,
                              mat.data(),
                              Dim,
                              ipiv.data(),
                              err.data(),
                              Dim);

    if (info != 0) {
      std::cout << info << std::endl;
      throw std::runtime_error("solveNewton() - LAPACKE_dgesv");
    }

    res[0] -= err(0);
    res[1] -= err(1);
    if (Dim == 3) {
      res[2] -= err(2);
    }

    vel = v(res, tn);
    err(0) = res[0] - rhs[0] - coef * vel[0];
    err(1) = res[1] - rhs[1] - coef * vel[1];
    if (Dim == 3)
      err(2) = res[2] - rhs[2] - coef * vel[2];
    if (i == 10) {
      throw std::runtime_error(
          "solveNewton() reach the maximal number of iterations");
    }
  }
  return res;
}

template <int Dim, RK::Type_Minor Type, template <int> class VectorRHS>
class DIRK : public TimeIntegrator<Dim, VectorRHS> {
  template <class T>
  using Vector = std::vector<T>;

  using Point = Vec<Real, Dim>;

  using ButcherTab = ButcherTableau<RK::DIRK, Type>;

  using SpMat = Eigen::SparseMatrix<Real>;

public:
  const int order = ButcherTab::order;

  const Point timeStep(const VectorRHS<Dim> &v,
                       const Point &pt,
                       Real tn,
                       Real dt);

  void timeStep(const VectorRHS<Dim> &v, Vector<Point> &pts, Real tn, Real dt);
};

template <int Dim, RK::Type_Minor Type>
class DIRK<Dim, Type, VectorFunction>
    : public TimeIntegrator<Dim, VectorFunction> {
  template <class T>
  using Vector = std::vector<T>;

  using Point = Vec<Real, Dim>;

  using ButcherTab = ButcherTableau<RK::DIRK, Type>;

public:
  const int order = ButcherTab::order;

  const Point timeStep(const VectorFunction<Dim> &v,
                       const Point &pt,
                       Real tn,
                       Real dt) {
    Vector<Point> step(ButcherTab::nStages);
    Point tmpt;
    Point result = pt;
    Vector<Real> ptjac = v.getJacobi(pt, tn);
    Tensor<Real, 2> jacobi(Dim);
    for (int j = 0; j < Dim; j++) {
      for (int k = 0; k < Dim; k++) {
        jacobi(j, k) = ptjac[j * Dim + k];
      }
    }

    for (int i = 0; i < ButcherTab::nStages; i++) {
      tmpt = pt;
      for (int j = 0; j < i; j++) {
        tmpt = tmpt + step[j] * ButcherTab::a[i][j] * dt;
      }
      if (ButcherTab::a[i][i] != 0) {
        tmpt = solveNewton<Dim>(v,
                                tn + ButcherTab::c[i] * dt,
                                ButcherTab::a[i][i] * dt,
                                jacobi,
                                tmpt,
                                pt,
                                1e-15);
      }
      step[i] = v(tmpt, tn + ButcherTab::c[i] * dt);
      result = result + step[i] * ButcherTab::b[i] * dt;
    }
    return result;
  }

  void timeStep(const VectorFunction<Dim> &v,
                Vector<Point> &pts,
                Real tn,
                Real dt) {
    int num = pts.size();
    for (int i = 0; i < num; i++) {
      pts[i] = this->timeStep(v, pts[i], tn, dt);
    }
  }
};

// VectorOnHypersurface version

template <int Dim>
std::vector<Vec<Real, Dim>> solveNewton(const VectorOnHypersurface<Dim> &v,
                                        Real tn,
                                        Real coef,
                                        const Tensor<Real, 2> &Jacobi,
                                        const std::vector<Vec<Real, Dim>> &rhs,
                                        const std::vector<Vec<Real, Dim>> &pts,
                                        Real tol) {
  using Point = Vec<Real, Dim>;

  int num = pts.size() - 1;
  auto res = pts;
  assert(Dim * num == Jacobi.size()[0]);

  Tensor<Real, 2> mat = -Jacobi * coef;
  for (int i = 0; i < Dim * num; i++) {
    mat(i, i) += 1;
  }

  // std::cout << mat << std::endl;

  std::vector<Point> vel;
  Tensor<Real, 1> err(Dim * num);
  Tensor<int, 1> ipiv(Dim * num);

  vel = v(res, tn);
  for (int j = 0; j < num; j++) {
    err(j) = res[j][0] - rhs[j][0] - coef * vel[j][0];
    err(j + num) = res[j][1] - rhs[j][1] - coef * vel[j][1];
    if (Dim == 3)
      err(j + 2 * num) = res[j][2] - rhs[j][2] - coef * vel[j][2];
  }

  for (int i = 0; i < 10; i++) {
    if (norm(err) < tol)
      break;
    auto info = LAPACKE_dgesv(LAPACK_COL_MAJOR,
                              2 * num,
                              1,
                              mat.data(),
                              2 * num,
                              ipiv.data(),
                              err.data(),
                              2 * num);
    if (info != 0) {
      std::cout << info << std::endl;
      throw std::runtime_error("solveNewton() - DGESV");
    }
    for (int j = 0; j < num; j++) {
      res[j][0] -= err(j);
      res[j][1] -= err(j + num);
      if (Dim == 3) {
        res[j][2] -= err(j + 2 * num);
      }
    }
    res[num][0] -= err(0);
    res[num][1] -= err(num);
    if (Dim == 3) {
      res[num][2] -= err(2 * num);
    }
    vel = v(res, tn);
    for (int j = 0; j < num; j++) {
      err(j) = res[j][0] - rhs[j][0] - coef * vel[j][0];
      err(j + num) = res[j][1] - rhs[j][1] - coef * vel[j][1];
      if (Dim == 3)
        err(j + 2 * num) = res[j][2] - rhs[j][2] - coef * vel[j][2];
    }
    if (i == 10) {
      exit(1);
    }
  }
  return res;
}

template <int Dim, RK::Type_Minor Type>
class DIRK<Dim, Type, VectorOnHypersurface>
    : public TimeIntegrator<Dim, VectorOnHypersurface> {
  template <class T>
  using Vector = std::vector<T>;

  using Point = Vec<Real, Dim>;

  using ButcherTab = ButcherTableau<RK::DIRK, Type>;

public:
  const int order = ButcherTab::order;

  const Point timeStep(const VectorOnHypersurface<Dim> &v,
                       const Point &pt,
                       Real tn,
                       Real dt) {
    throw pt;
  }

  void timeStep(const VectorOnHypersurface<Dim> &v,
                Vector<Point> &pts,
                Real tn,
                Real dt) {
    int num = pts.size();
    // Tensor<Real, 2> jacobi = v.getJacobi(pts, tn);
    auto jacobi = v.getJacobi(pts, tn);
    Vector<Vector<Point>> step;
    step.resize(ButcherTab::nStages);
    Vector<Point> tmpts;
    Vector<Point> result = pts;

    for (int i = 0; i < ButcherTab::nStages; i++) {
      tmpts = pts;
      for (int j = 0; j < i; j++) {
        for (int k = 0; k < num; k++) {
          tmpts[k] = tmpts[k] + step[j][k] * ButcherTab::a[i][j] * dt;
        }
      }
      if (ButcherTab::a[i][i] != 0) {
        tmpts = solveNewton<Dim>(v,
                                 tn + ButcherTab::c[i] * dt,
                                 ButcherTab::a[i][i] * dt,
                                 jacobi,
                                 tmpts,
                                 pts,
                                 1e-10);
      }
      step[i] = v(tmpts, tn + ButcherTab::c[i] * dt);
      for (int k = 0; k < num; k++) {
        result[k] = result[k] + step[i][k] * ButcherTab::b[i] * dt;
      }
    }
    pts = result;
  }
};

#endif
