#ifndef ENERGYFUNCTOR_H
#define ENERGYFUNCTOR_H
#include <cmath>
template <class T>
class EnergyFunctor {
public:
  virtual T calculateEnergy(T dist, T R) = 0;
  virtual T calculateEnergyDerivative(T dist, T R) = 0;
};

template <class T>
class LinearEnergyFunctor : public EnergyFunctor<T> {
public:
  T calculateEnergy(T dist, T R) { return std::pow(dist - R, 2) / 2; }
  T calculateEnergyDerivative(T dist, T R) { return dist - R; }
};

template <class T>
class LogarithmEnergyFunctor : public EnergyFunctor<T> {
public:
  T calculateEnergy(T dist, T R) { return dist * log(dist / R) - dist + R; }
  T calculateEnergyDerivative(T dist, T R) { return log(dist / R); }
};

#endif