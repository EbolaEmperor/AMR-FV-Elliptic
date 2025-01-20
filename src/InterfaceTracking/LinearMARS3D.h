#ifndef _LINEARMARS3D_H_
#define _LINEARMARS3D_H_

#include "MARS.h"
#include "PolynomialSurface.h"
#include "YinSet3D/TriangleCompare.h"
#include "YinSet3D/YinSet.h"

template <int Order, template <int> class VelocityField>
class LinearMARS3D : public MARS<3, 2, VelocityField> {
  using Base = MARS<3, 2, VelocityField>;

public:
  LinearMARS3D() = delete;

  LinearMARS3D(TimeIntegrator<3, VelocityField> *_TI,
               Real hL,
               Real _minAngle,
               Real rtiny = 0.1) :
      Base(_TI),
      chdLenRange(Vec<Real, 1>(rtiny * hL), Vec<Real, 1>(hL)),
      minAngle(_minAngle) {}
  void timeStep(const VelocityField<3> &v, YinSet<3, 2> &ys, Real tn, Real dt);

private:
  void discreteFlowMap(const VelocityField<3> &v,
                       YSB::YinSet<3, 2> &ys,
                       Real tn,
                       Real k);
  void splitLongEdges(const VelocityField<3> &v,
                      YSB::YinSet<3, 2> &ys,
                      const YSB::YinSet<3, 2> &ystn,
                      Real tn,
                      Real k);
  void splitLongEdges(
      const VelocityField<3> &v,
      YSB::Triangle<Real, 3> &tri,
      YSB::Triangle<Real, 3> &tritn,
      std::map<YSB::Triangle<Real, 3>, bool, YSB::TriangleCompare> &record,
      Real tn,
      Real k);
  void removeSmallEdges(YSB::YinSet<3, 2> &ys);
  void swapEdges(YSB::YinSet<3, 2> &ys);
  void adjustVertex(YSB::YinSet<3, 2> &ys);
  void regenerate(YSB::YinSet<3, 2> &ys);

private:
  Real minAngle;
  Interval<1> chdLenRange;
};

template <int Order>
class LinearMARS3D<Order, VectorFunction> : public MARS<3, 2, VectorFunction> {
  using Base = MARS<3, 2, VectorFunction>;

public:
  LinearMARS3D() = delete;

  LinearMARS3D(TimeIntegrator<3, VectorFunction> *_TI,
               Real hL,
               Real _minAngle,
               Real rtiny = 0.1) :
      Base(_TI),
      chdLenRange(Vec<Real, 1>(rtiny * hL), Vec<Real, 1>(hL)),
      minAngle(_minAngle) {}
  void timeStep(const VectorFunction<3> &v,
                YSB::YinSet<3, 2> &ys,
                Real tn,
                Real dt);
  void timeStep(const VectorFunction<3> &v,
                YSB::YinSet<3, 2> &ys,
                Real tn,
                Real dt,
                int *numProcessed);

private:
  void discreteFlowMap(const VectorFunction<3> &v,
                       YSB::YinSet<3, 2> &ys,
                       Real tn,
                       Real k);
  void splitLongEdges(const VectorFunction<3> &v,
                      YSB::YinSet<3, 2> &ys,
                      const YSB::YinSet<3, 2> &ystn,
                      Real tn,
                      Real k);
  void splitLongEdges(
      const VectorFunction<3> &v,
      YSB::Triangle<Real, 3> &tri,
      YSB::Triangle<Real, 3> &tritn,
      std::map<YSB::Triangle<Real, 3>, bool, YSB::TriangleCompare> &record,
      Real tn,
      Real k);
  void splitLongEdgesHighOrder(
      const VectorFunction<3> &v,
      YSB::Triangle<Real, 3> &tri,
      YSB::Triangle<Real, 3> &tritn,
      YSB::Triangle<Real, 3> &triOriginaltn,
      std::map<YSB::Triangle<Real, 3>, bool, YSB::TriangleCompare> &record,
      const PolynomialSurface<2> &triPolySurf,
      std::map<YSB::Segment<Real, 3>,
               PolynomialSurface<2>,
               YSB::SegmentCompare> &edgePolySurf,
      Real tn,
      Real k);
  void removeSmallEdges(YSB::YinSet<3, 2> &ys);
  void swapEdges(YSB::YinSet<3, 2> &ys);
  void adjustVertex(YSB::YinSet<3, 2> &ys);
  void regenerate(YSB::YinSet<3, 2> &ys, int wrongid);
  bool isRegular(YSB::SurfacePatch<Real, 2> &sp);
  bool isOriented(YSB::YinSet<3, 2> &ys);
  int countSmallAngle(const std::vector<YSB::Triangle<Real, 3>> &vecTri);

private:
  Real minAngle;
  Interval<1> chdLenRange;
};

#endif