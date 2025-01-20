#pragma once

#include "AMRTools/Utilities.h"
#include "FiniteDiff/FuncFiller.h"

#include <map>
#include <string>

//==========================================================
/**
 * The user interface of a function object.
 * @tparam Dim The dimension of the domain of the function.
 */
template <int Dim>
class FunctionWrapper {
public:
  Real t = 0.0;  ///< The time instant

  virtual ~FunctionWrapper() = default;

  virtual Real operator()(const Vec<Real, Dim> &) const { return 0.; };

  // Some member methods for vector-value functions
  virtual Real operator()(const Vec<Real, Dim> &, int comp) const {
    return 0.;
  };

  virtual Vec<Real, Dim> vectorValue(const Vec<Real, Dim> &x) const {
    Vec<Real, Dim> ans;
    for (int d = 0; d < Dim; ++d)
      ans[d] = (*this)(x, d);
    return ans;
  };

  virtual Real dot(const Vec<Real, Dim> &x, const Vec<Real, Dim> &dir) const {
    Real ans = 0;
    for (int d = 0; d < Dim; ++d)
      if (dir[d] != 0)
        ans += (*this)(x, d) * dir[d];
    return ans;
  };
};

//==========================================================

// a forward declaration
template <int Dim>
class FunctionFactory;

/**
 * A lookup table (singleton)
 * for all the functions of 'Dim' dimensions.
 */
template <int Dim>
class FunctionTable {
public:
  static FunctionTable &getInstance() {
    static FunctionTable singleton;
    return singleton;
  }

  FunctionTable();

protected:
  using Alloc = FunctionWrapper<Dim> *(*)(const lightJSON::jsonNode &);
  std::map<std::string, Alloc> allocTable;
  friend class FunctionFactory<Dim>;
};

//==========================================================
/**
 * The factory for generating functions
 * from a string.
 */
template <int Dim>
class FunctionFactory {
public:
  using FWR = FunctionWrapper<Dim>;

  FunctionFactory() {}

  FWR *getFunc(const lightJSON::jsonNode &nFunc) const;

  void deleteFunc(FWR *pFunc) const { delete pFunc; }
};
