#ifndef EDGESWAPPER_H
#define EDGESWAPPER_H

#include "YinSet3D/YinSet.h"

class EdgeSwapper {
public:
  EdgeSwapper(Real hL, Real _minAngle, Real rtiny = 0.1) :
      chdLenRange(Vec<Real, 1>(rtiny * hL), Vec<Real, 1>(hL)),
      minAngle(_minAngle) {}

  void swap(YSB::SurfacePatch<Real, 2> &sp);
  void swapLocally(YSB::YinSet<3, 2> &ys);
  void swapGlobally(YSB::YinSet<3, 2> &ys);

private:
  Real minAngle;
  Interval<1> chdLenRange;
};

#endif