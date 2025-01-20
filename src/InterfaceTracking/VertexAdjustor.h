#ifndef VERTEXADJUSTOR_H
#define VERTEXADJUSTOR_H

#include "InterfaceTracking/EnergyFunctor.h"
#include "YinSet3D/GluedSurface.h"
#include "YinSet3D/SurfacePatch.h"

class Polygon;

template <int Order>
class VertexAdjustor {
  EnergyFunctor<Real> *EF;
  Interval<1> chdLenRange;
  Real minAngle;

  int maxLocalIter;   // maximum iterations of the gradient descent for local
                      // adjustment
  int maxGlobalIter;  // maximum iterations of the gradient descent for global
                      // adjustment
  Real c;             // coefficient for Armijo condition
  Real rho;           // coefficient for backtracking
  Real alpha;         // coefficient for restricting the step size

public:
  VertexAdjustor(EnergyFunctor<Real> *_EF,
                 Interval<1> _chdLenRange,
                 Real _minAngle,
                 int _maxLocalIter = 10,
                 int _maxGlobalIter = 100,
                 Real _c = 1e-4,
                 Real _rho = 0.8,
                 Real _alpha = 0.3) :
      EF(_EF),
      chdLenRange(_chdLenRange),
      minAngle(_minAngle),
      maxLocalIter(_maxLocalIter),
      maxGlobalIter(_maxGlobalIter),
      c(_c),
      rho(_rho),
      alpha(_alpha) {}

  bool adjustLocally(YSB::SurfacePatch<Real, 2> &sp, Real R);
  void projectOnTriangulation(
      const std::vector<YSB::Triangle<Real, 3>> &vecTri,
      YSB::Point<Real, 3> &ap);
  void adjustLocally2D(const Polygon &pg,
                       std::vector<Vec<Real, 2>> &adjustedPts,
                       Real R);
  void regenerateSurfacePatch(
      YSB::SurfacePatch<Real, 2> &sp,
      std::set<YSB::Point<Real, 3>, YSB::PointCompare> setTempP,
      int wrongid);

private:
  Real minVecTriAngle(const std::vector<YSB::Triangle<Real, 3>> &vecTri);
  Real minVecTriAngle2D(const std::vector<YSB::Triangle<Real, 2>> &vecTri);
};

#endif