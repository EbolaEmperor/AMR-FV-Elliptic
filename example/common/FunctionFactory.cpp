#include "FunctionFactory.h"

#include "Core/Vec.h"
#include "FiniteDiff/FuncFiller.h"

#include <vector>

//============================================================
// Below goes the pre-defined functions
namespace NS4_Functions {

template <int Dim>
class Constant : public FunctionWrapper<Dim> {
public:
  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new Constant<Dim>(js);
  }

  Constant(Real c = 0) : c_(c) {}
  Constant(const lightJSON::jsonNode &js) {
    Real constant;
    PGET(js, constant);
    c_ = constant;
  }

  Real operator()(const Vec<Real, Dim> &) const { return c_; }
  Real operator()(const Vec<Real, Dim> &, int) const { return c_; }

private:
  Real c_;
};

template <int Dim>
class InvSum : public FunctionWrapper<Dim> {
public:
  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &) {
    return new InvSum<Dim>();
  }

  Real operator()(const Vec<Real, Dim> &x) const {
    Real sum = 0.;
    for (int i = 0; i < Dim; ++i)
      sum += x[i];
    return 1. / sum;
  }
};

template <int Dim>
class Mollifier : public FunctionWrapper<Dim> {
public:
  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new Mollifier<Dim>(js);
  }

  Mollifier(Vec<Real, Dim> center = 0.5,
            Real radius = 0.1,
            Real maxValue = 1) :
      center(center), radius(radius), maxValue(maxValue) {}

  Mollifier(const lightJSON::jsonNode &js) {
    PGET(js, center);
    PGET(js, radius);
    PGET(js, maxValue);
    maxValue *= exp(1);
  }

  Real stdMollifier(const Vec<Real, Dim> &x) const {
    return (dot(x, x) < 1) ? maxValue * exp(1. / (dot(x, x) - 1.)) : 0;
  }

  Real operator()(const Vec<Real, Dim> &x) const {
    return stdMollifier((x - center) / radius);
  }

private:
  Vec<Real, Dim> center;
  Real radius;
  Real maxValue;
};

template <int Dim>
class GaussianBase : public FunctionWrapper<Dim> {
public:
  GaussianBase() {}

  GaussianBase(Vec<Real, Dim> center, Real lamb) {
    centers_.push_back(center);
    lambs_.push_back(lamb);
  }

  GaussianBase(const std::vector<Vec<Real, Dim>> &centers,
               const std::vector<Real> &lambs) :
      centers_(centers), lambs_(lambs) {}

  GaussianBase(const lightJSON::jsonNode &func) {
    std::vector<Vec<Real, Dim>> centers;
    std::vector<Real> lambs;
    PGET(func, centers);
    PGET(func, lambs);
    std::swap(centers_, centers);
    std::swap(lambs_, lambs);
  }

protected:
  std::vector<Vec<Real, Dim>> centers_;
  std::vector<Real> lambs_;
};

template <int Dim>
class Gaussian : public GaussianBase<Dim> {
public:
  using GaussianBase<Dim>::GaussianBase;
  using GaussianBase<Dim>::centers_;
  using GaussianBase<Dim>::lambs_;

  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new Gaussian<Dim>(js);
  }

  Real operator()(const Vec<Real, Dim> &x) const {
    Real ans = 0.;
    for (unsigned i = 0; i < centers_.size(); ++i)
      ans += exp(-lambs_[i] * pow(norm(x - centers_[i], 2), 2));
    return ans;
  }
};

template <int Dim>
class GaussianLap : public GaussianBase<Dim> {
public:
  using GaussianBase<Dim>::GaussianBase;
  using GaussianBase<Dim>::centers_;
  using GaussianBase<Dim>::lambs_;

  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new GaussianLap<Dim>(js);
  }

  Real operator()(const Vec<Real, Dim> &x) const {
    Real ans = 0.;
    for (unsigned i = 0; i < centers_.size(); ++i)
      ans += 4 * lambs_[i] *
             (lambs_[i] * pow(norm(x - centers_[i], 2), 2) - 1) *
             exp(-lambs_[i] * pow(norm(x - centers_[i], 2), 2));
    return ans;
  }
};

template <int Dim>
class GaussianGrad : public GaussianBase<Dim> {
public:
  using GaussianBase<Dim>::GaussianBase;
  using GaussianBase<Dim>::centers_;
  using GaussianBase<Dim>::lambs_;

  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new GaussianGrad<Dim>(js);
  }

  Real operator()(const Vec<Real, Dim> &x, int comp) const {
    assert(0 <= comp && comp < Dim);
    Real ans = 0.;
    for (unsigned i = 0; i < centers_.size(); ++i)
      ans += -2. * lambs_[i] * (x[comp] - centers_[i][comp]) *
             exp(-lambs_[i] * pow(norm(x - centers_[i], 2), 2));
    return ans;
  }
};

template <int Dim>
class SinMulti : public FunctionWrapper<Dim> {
public:
  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new SinMulti<Dim>();
  }
  Real operator()(const Vec<Real, Dim> &x) const {
    Real ans = 1.;
    for (int d = 0; d < Dim; ++d)
      ans *= sin(M_PI * x[d]);
    return ans;
  }
};

template <int Dim>
class SinMultiLap : public FunctionWrapper<Dim> {
public:
  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new SinMultiLap<Dim>();
  }
  Real operator()(const Vec<Real, Dim> &x) const {
    Real ans = -M_PI * M_PI * Dim;
    for (int d = 0; d < Dim; ++d)
      ans *= sin(M_PI * x[d]);
    return ans;
  }
};

template <int Dim>
class SinMultiGrad : public FunctionWrapper<Dim> {
public:
  static FunctionWrapper<Dim> *alloc(const lightJSON::jsonNode &js) {
    return new SinMultiGrad<Dim>();
  }
  Real operator()(const Vec<Real, Dim> &x, int comp) const {
    Real ans = M_PI;
    for (int d = 0; d < Dim; ++d)
      ans *= (d == comp) ? cos(M_PI * x[d]) : sin(M_PI * x[d]);
    return ans;
  }
};

}  // namespace NS4_Functions

//============================================================
// Below are the registrations of the pre-defined functions
template <>
FunctionTable<2>::FunctionTable() {
#define ADDFUNC(name, classname)                                              \
  allocTable.insert(                                                          \
      std::make_pair(std::string(name), NS4_Functions::classname::alloc))

  ADDFUNC("Gaussian", Gaussian<2>);
  ADDFUNC("Gaussian_Lap", GaussianLap<2>);
  ADDFUNC("Gaussian_Grad", GaussianGrad<2>);
  ADDFUNC("SinSin", SinMulti<2>);
  ADDFUNC("SinSin_Lap", SinMultiLap<2>);
  ADDFUNC("SinSin_Grad", SinMultiGrad<2>);
  ADDFUNC("Constant", Constant<2>);
  ADDFUNC("Mollifier", Mollifier<2>);
  ADDFUNC("InvSum", InvSum<2>);

#undef ADDFUNC
}

template <int Dim>
FunctionWrapper<Dim> *FunctionFactory<Dim>::getFunc(
    const lightJSON::jsonNode &nFunc) const {
  const auto &table = FunctionTable<Dim>::getInstance();
  std::string description;
  PGET(nFunc, description);
  auto it = table.allocTable.find(description);
  assert(it != table.allocTable.cend());
  return (it->second)(nFunc);
}

template class FunctionFactory<2>;

template void FuncFiller<2>::fillAvr(LevelData<Real, 2> &,
                                     const FunctionWrapper<2> &,
                                     bool,
                                     unsigned) const;

template void FuncFiller<2>::fillBdryAvr(Vector<Array<Tensor<Real, 1>, 4>> &,
                                         const FunctionWrapper<2> &,
                                         int,
                                         int,
                                         int) const;

template void FuncFiller<2>::fillBdryAvr(Vector<Array<Tensor<Real, 1>, 4>> &,
                                         const FunctionWrapper<2> &,
                                         int) const;