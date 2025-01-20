#ifndef _EXPLICIT_RK_H_
#define _EXPLICIT_RK_H_

#include "RKButcher.h"
#include "TimeIntegrator.h"

#include <vector>

/// The compile time constants of numbers of stages

template <int Dim, RK::Type_Minor Type, template <int> class VectorRHS>
class ERK : public TimeIntegrator<Dim, VectorRHS> {
  template <class T>
  using Vector = std::vector<T>;

  using Point = Vec<Real, Dim>;

  using ButcherTab = ButcherTableau<RK::ERK, Type>;

  // enum{order = ButcherTab::nStages};

public:
  const int order = ButcherTab::order;

  const Point timeStep(const VectorRHS<Dim> &v,
                       const Point &pt,
                       Real tn,
                       Real dt);

  void timeStep(const VectorRHS<Dim> &v, Vector<Point> &pts, Real tn, Real dt);
};

// VectorFunction version

template <int Dim, RK::Type_Minor Type>
class ERK<Dim, Type, VectorFunction>
    : public TimeIntegrator<Dim, VectorFunction> {
  template <class T>
  using Vector = std::vector<T>;

  using Point = Vec<Real, Dim>;

  using ButcherTab = ButcherTableau<RK::ERK, Type>;

  // enum{order = ButcherTab::nStages};

public:
  const int order = ButcherTab::order;

  const Point timeStep(const VectorFunction<Dim> &v,
                       const Point &pt,
                       Real tn,
                       Real dt) {
    Vector<Point> step;
    step.resize(ButcherTab::nStages);
    Point tmpt;
    Point result = pt;

    for (int i = 0; i < ButcherTab::nStages; i++) {
      tmpt = pt;
      for (int j = 0; j < i; j++) {
        tmpt = tmpt + step[j] * ButcherTab::a[i][j] * dt;
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
      step[i] = v(tmpts, tn + ButcherTab::c[i] * dt);
      for (int k = 0; k < num; k++) {
        result[k] = result[k] + step[i][k] * ButcherTab::b[i] * dt;
      }
    }
    pts = result;
  }
};

// VectorOnHypersurface version

// template <int Dim, RK::Type_Minor Type>
// class ERK<Dim, Type, VectorOnHypersurface> : public TimeIntegrator<Dim,
// VectorOnHypersurface>
// {

//     template <class T>
//     using Vector = std::vector<T>;

//     using Point = Vec<Real, Dim>;

//     using ButcherTab = ButcherTableau<RK::ERK, Type>;

//     //enum{order = ButcherTab::nStages};

// public:
//     const int order = ButcherTab::order;

//     const Point timeStep(const VectorOnHypersurface<Dim> &v, const Point
//     &pt, Real tn, Real dt)
//     {
//       throw pt;
//     }

//     void timeStep(const VectorOnHypersurface<Dim> &v, Vector<Point> &pts,
//     Real tn, Real dt)
//     {
//         int num = pts.size();
//         Vector<Vector<Point>> step;
//         step.resize(ButcherTab::nStages);
//         Vector<Point> tmpts;
//         Vector<Point> result = pts;

//         for (int i = 0; i < ButcherTab::nStages; i++)
//         {
//             tmpts = pts;
//             for (int j = 0; j < i; j++)
//             {
//                 for (int k = 0; k < num; k++)
//                 {
//                     tmpts[k] = tmpts[k] + step[j][k] * ButcherTab::a[i][j] *
//                     dt;
//                 }
//             }
//             step[i] = v(tmpts, tn + ButcherTab::c[i] * dt);
//             for (int k = 0; k < num; k++)
//             {
//                 result[k] = result[k] + step[i][k] * ButcherTab::b[i] * dt;
//             }
//         }
//         pts = result;
//     }
// };

#endif
