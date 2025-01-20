#ifndef _MARS2D_H_
#define _MARS2D_H_

#include "Core/Curve.h"
#include "Core/Interval.h"
#include "MARS.h"

#include <vector>

template <int Order, template <int> class VelocityField>
class MARS2D : public MARS<2, Order, VelocityField> {
  template <class T>
  using Vector = std::vector<T>;

  using YS = YinSet<2, Order>;

  using Point = Vec<Real, 2>;

  using Crv = Curve<2, Order>;

  using Base = MARS<2, Order, VelocityField>;

public:
  MARS2D() = delete;

  MARS2D(TimeIntegrator<2, VelocityField> *_TI, Real hL, Real rtiny = 0.1) :
      Base(_TI), chdLenRange(Vec<Real, 1>(rtiny * hL), Vec<Real, 1>(hL)) {}

  void timeStep(const VelocityField<2> &v, YS &ys, Real tn, Real dt);

private:
  void discreteFlowMap(const VelocityField<2> &v,
                       Vector<Point> &pts,
                       Real tn,
                       Real dt);

  Vector<unsigned int> removeSmallEdges(Vector<Point> &pts);

  Vector<unsigned int> splitLongEdges(const VelocityField<2> &v,
                                      Vector<Point> &pts,
                                      const Crv &crv,
                                      Real tn,
                                      Real dt);

private:
  Interval<1> chdLenRange;
};

// partial specialization for VectorFunction
template <int Order>
class MARS2D<Order, VectorFunction> : public MARS<2, Order, VectorFunction> {
  template <class T>
  using Vector = std::vector<T>;

  using YS = YinSet<2, Order>;

  using Point = Vec<Real, 2>;

  using Crv = Curve<2, Order>;

  using Base = MARS<2, Order, VectorFunction>;

public:
  MARS2D() = delete;

  MARS2D(TimeIntegrator<2, VectorFunction> *_TI, Real hL, Real rtiny = 0.1) :
      Base(_TI), chdLenRange(Vec<Real, 1>(rtiny * hL), Vec<Real, 1>(hL)) {}

  void timeStep(const VectorFunction<2> &v, YS &ys, Real tn, Real dt);

private:
  void discreteFlowMap(const VectorFunction<2> &v,
                       Vector<Point> &pts,
                       Real tn,
                       Real dt);

  Vector<unsigned int> removeSmallEdges(Vector<Point> &pts);

  Vector<unsigned int> splitLongEdges(const VectorFunction<2> &v,
                                      Vector<Point> &pts,
                                      const Crv &crv,
                                      Real tn,
                                      Real dt);

private:
  Interval<1> chdLenRange;
};

// partial specialization for VectorOnHypersurface
template <int Order>
class MARS2D<Order, VectorOnHypersurface>
    : public MARS<2, Order, VectorOnHypersurface> {
  template <class T>
  using Vector = std::vector<T>;

  using YS = YinSet<2, Order>;

  using Point = Vec<Real, 2>;

  using Crv = Curve<2, Order>;

  using Base = MARS<2, Order, VectorOnHypersurface>;

public:
  MARS2D() = delete;

  MARS2D(TimeIntegrator<2, VectorOnHypersurface> *_TI,
         Real hL,
         Real rtiny = 0.1) :
      Base(_TI), chdLenRange(Vec<Real, 1>(rtiny * hL), Vec<Real, 1>(hL)) {}

  void timeStep(const VectorOnHypersurface<2> &v, YS &ys, Real tn, Real dt);

private:
  void discreteFlowMap(const VectorOnHypersurface<2> &v,
                       Vector<Point> &pts,
                       Real tn,
                       Real dt);

  Vector<unsigned int> removeSmallEdges(Vector<Point> &pts);

  Vector<unsigned int> splitLongEdges(const VectorOnHypersurface<2> &v,
                                      Vector<Point> &pts,
                                      const Crv &crv,
                                      Real tn,
                                      Real dt);

private:
  Interval<1> chdLenRange;
};

#endif
