#ifndef MULTIVARIATEPOLYNOMIAL_H
#define MULTIVARIATEPOLYNOMIAL_H

#include "Core/Config.h"
#include "Core/numlib.h"

/// Represent a degree Ndeg polynomial of Nvar variables.
/**
 * @See https://github.com/uekstrom/polymul/blob/master/polymul.h
 */
template <int Nvar, int Ndeg, class CoefType = Real>
class MultivariatePolynomial;

// this helper takes care of the reduction of order when finding derivatives
template <int Ndeg, int Var>
struct _diff_wrapper_2D;

// this helper takes care of the evaluation of a multivariate polynomial
template <int Ndeg, class CoefType>
struct _eval_wrapper_2D;

///  Template specialization
/**
 * Represent a bivariate polynomial
 */
template <int Ndeg, class CoefType>
class MultivariatePolynomial<2, Ndeg, CoefType> {
public:
  enum { Dim = 2, Order = Ndeg + 1 };
  enum { nBasis = binom(Ndeg + Dim, Dim) };

public:
  /**
   * Construct a polynomial with constant coefficient c0
   *  and other terms zero.
   */
  MultivariatePolynomial(const CoefType &c0 = CoefType()) {
    coefs[0][0] = c0;
    for (int i = 1; i <= Ndeg; ++i)
      for (int j = 0; j <= Ndeg - i; ++j)
        coefs[i][j] = CoefType();
  }

  /**
   * Construct a polynomial according to the given coefficients
   *  ordered by graded lexicographical, for example 1 x y x^2 xy y^2 ..
   */
  MultivariatePolynomial(const std::initializer_list<CoefType> l) {
    auto q = l.begin();
    for (int deg = 0; deg <= Ndeg; ++deg)
      for (int i = deg; i >= 0; --i)
        coefs[i][deg - i] = *q++;
  }

public:
  /**
   * return coefficient c[i][j]
   */
  const CoefType &coefficient(int i, int j) const { return coefs[i][j]; }

  CoefType &coefficient(int i, int j) { return coefs[i][j]; }

public:
  /**
   * Evaluate the polynomial at a point p.
   */
  template <class T>
  CoefType operator()(const Vec<T, Dim> &p) const {
    return _eval_wrapper_2D<Ndeg, CoefType>::calc(*this, p);
  }

  /**
   * Calculate the derivative of this with respect to variable var.
   */
  template <int Var>
  auto diff() const {
    return _diff_wrapper_2D<Ndeg, Var>::calc(*this);
  }

public:
  /**
   * Arithmetic
   */
  MultivariatePolynomial<Dim, Ndeg, CoefType> operator+(
      const MultivariatePolynomial<Dim, Ndeg, CoefType> &rhs) const;

  MultivariatePolynomial<Dim, Ndeg, CoefType> operator-(
      const MultivariatePolynomial<Dim, Ndeg, CoefType> &rhs) const;

  template <int Ndeg1, int Ndeg2, class CT>
  friend MultivariatePolynomial<Dim, Ndeg1 + Ndeg2, CT> operator*(
      const MultivariatePolynomial<Dim, Ndeg1, CT> &lhs,
      const MultivariatePolynomial<Dim, Ndeg2, CT> &rhs);

public:
  /**
   * Output
   */
  friend std::ostream &operator<<(
      std::ostream &os,
      const MultivariatePolynomial<Dim, Ndeg, CoefType> &mp) {
    os << "A bivariate polynomial of degree " << Ndeg
       << ", with the coefficients :\n";
    for (int i = 0; i <= Ndeg; ++i) {
      for (int j = 0; j <= Ndeg - i; ++j)
        os << mp.coefs[i][j] << ", ";
      for (int j = Ndeg - i + 1; j <= Ndeg; ++j)
        os << "x, ";
      os << "\n";
    }
    return os;
  }

protected:
  /**
   * A bivariate polynomial has the form
   *   p(x) = \sum_{i+j<=Ndeg} c[i][j]*x^iy^j.
   */
  CoefType coefs[Order][Order];
};

template <int Ndeg, class CoefType>
struct _eval_wrapper_2D {
  template <class T>
  static CoefType calc(const MultivariatePolynomial<2, Ndeg, CoefType> &mp,
                       const Vec<T, 2> &p) {
    // use Horner's scheme over coef[i]
    CoefType res = static_cast<CoefType>(0.0);
    T basis = static_cast<T>(1.0);
    for (int i = 0; i <= Ndeg; ++i) {
      CoefType res_i = static_cast<CoefType>(0.0);
      for (int j = Ndeg - i; j >= 0; --j) {
        res_i = res_i * p[1] + mp.coefficient(i, j);
      }
      res += res_i * basis;
      basis *= p[0];
    }
    return res;
  }
};

template <int Ndeg, int Var>
struct _diff_wrapper_2D {
  template <class CoefType>
  static MultivariatePolynomial<2, Ndeg - 1, CoefType> calc(
      const MultivariatePolynomial<2, Ndeg, CoefType> &mp) {
    MultivariatePolynomial<2, Ndeg - 1, CoefType> res;
    for (int i = 0; i <= Ndeg; ++i)
      for (int j = 0; j <= Ndeg - i; ++j)
        /*        if constexpr(Var == 1)
                  res.coefficient(i,j) = (i+1) * mp.coefficient(i+1,
           j); else res.coefficient(i,j) = (j+1) * mp.coefficient(i,
           j+1);*/
        res.coefficient(i, j) = (Var == 1)
                                    ? (i + 1) * mp.coefficient(i + 1, j)
                                    : (j + 1) * mp.coefficient(i, j + 1);
    return res;
  }
};

template <int Var>
struct _diff_wrapper_2D<0, Var> {
  template <class CoefType>
  static MultivariatePolynomial<2, 0, CoefType> calc(
      const MultivariatePolynomial<2, 0, CoefType> &mp0) {
    return MultivariatePolynomial<2, 0, CoefType>(0);
  }
};

#endif  //　MULTIVARIATEPOLYNOMIAL_H
