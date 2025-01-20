#ifndef _MARS_H_
#define _MARS_H_

#include "TimeIntegrator.h"
#include "YinSet/YinSet.h"
#include "YinSet3D/YinSet.h"

#include <string>

template <int Dim, int Order, template <int> class VelocityField>
class MARS;

template <int Order, template <int> class VelocityField>
class MARS<2, Order, VelocityField> {
  using YS = YinSet<2, Order>;

protected:
  TimeIntegrator<2, VelocityField> *TI;

public:
  MARS(TimeIntegrator<2, VelocityField> *_TI) : TI(_TI) {}

  virtual void timeStep(const VelocityField<2> &v,
                        YS &ys,
                        Real tn,
                        Real dt) = 0;

  virtual void trackInterface(const VelocityField<2> &v,
                              YS &ys,
                              Real StartTime,
                              Real dt,
                              Real EndTime,
                              bool output = false,
                              std::string fName = "",
                              int opstride = 20);
};

template <int Order, template <int> class VelocityField>
class MARS<3, Order, VelocityField> {
  using YS = YSB::YinSet<3, Order>;

protected:
  TimeIntegrator<3, VelocityField> *TI;

public:
  MARS(TimeIntegrator<3, VelocityField> *_TI) : TI(_TI) {}

  virtual void timeStep(const VelocityField<3> &v,
                        YS &ys,
                        Real tn,
                        Real dt) = 0;

  virtual void trackInterface(const VelocityField<3> &v,
                              YS &ys,
                              Real StartTime,
                              Real dt,
                              Real EndTime,
                              bool output = false,
                              std::string fName = "",
                              int opstride = 20);
};

#endif