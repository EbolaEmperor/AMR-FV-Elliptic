#ifndef BINARYPOLYNOMIAL_H
#define BINARYPOLYNOMIAL_H

#include "Core/numlib.h"
// represent a binary polynomial.
template <int Order, class T>
class BinaryPolynomial {
public:
  enum { CoefNum = factorial(Order + 2) / (factorial(Order) * 2) };

private:
  // the coefficients are as follows: x^n x^(n-1)y ... y^n x^(n-1) x^(n-2)y ...
  // y^(n-1) ... x y 1.
  T coefs[CoefNum];

public:
  BinaryPolynomial(const T _c = T()) {
    for (int i = 0; i < CoefNum; i++)
      coefs[i] = _c;
  }
  BinaryPolynomial(std::initializer_list<T> _coefs) {
    auto q = _coefs.begin();
    for (int i = 0; i < CoefNum; i++)
      coefs[i] = *q++;
  }

  T &operator[](int k) { return coefs[k]; }
  const T &operator[](int k) const { return coefs[k]; }

  T operator()(T x, T y) const {
    T res = 0;
    int count = 0;
    for (int i = Order; i >= 0; i--) {
      for (int j = i; j >= 0; j--) {
        res += coefs[count++] * pow(x, j) * pow(y, i - j);
      }
    }
    return res;
  }
};

#endif