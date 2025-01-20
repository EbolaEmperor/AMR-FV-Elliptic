#ifndef _TIMEINTEGRATOR_H_
#define _TIMEINTEGRATOR_H_

#include "Core/Config.h"
#include "Core/Vec.h"
#include "VectorFunction.h"
//#include "GeometricFlow/VectorOnHypersurface.h"
#include <vector>

template <int Dim, template <int> class VectorRHS>
class TimeIntegrator {
  template <class T>
  using Vector = std::vector<T>;

  using Point = Vec<Real, Dim>;

public:
  virtual ~TimeIntegrator() {}

  virtual const Point timeStep(const VectorRHS<Dim> &v,
                               const Point &pt,
                               Real tn,
                               Real k) = 0;

  virtual void timeStep(const VectorRHS<Dim> &v,
                        Vector<Point> &pts,
                        Real tn,
                        Real k) = 0;
};

#endif
