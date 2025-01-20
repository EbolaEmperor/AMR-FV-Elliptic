#pragma once

template <int n, int k>
struct Factorial {
  enum { val = n * Factorial<n - 1, k - 1>::val };
};

template <int n>
struct Factorial<n, 1> {
  enum { val = n };
};

template <int n, int k>
struct BinomialCoefficient {
  enum { val = Factorial<n, k>::val / Factorial<k, k>::val };
};

template <int Dim, int Order, int RefRatio, bool IsGhost, int nAdd>
struct CFInterpStencilConst {
public:
  enum { nSrc = BinomialCoefficient<Order + Dim, Dim>::val + nAdd };
  enum { nDst = IsGhost ? 2 * RefRatio : RefRatio * RefRatio };
};

template <int Dim, int Order, int RefRatio, bool IsGhost, int nAdd>
class CFInterpStencil {
public:
  static const int
      srcpos[CFInterpStencilConst<Dim, Order, RefRatio, IsGhost, nAdd>::nSrc]
            [2];
  static const int
      dstpos[CFInterpStencilConst<Dim, Order, RefRatio, IsGhost, nAdd>::nDst]
            [2];
  static const double
      coef[CFInterpStencilConst<Dim, Order, RefRatio, IsGhost, nAdd>::nDst]
          [CFInterpStencilConst<Dim, Order, RefRatio, IsGhost, nAdd>::nSrc];

  typedef const int (*PointArr)[2];
  typedef const double (*CoefArr)
      [CFInterpStencilConst<Dim, Order, RefRatio, IsGhost, nAdd>::nSrc];

  static void getArr(PointArr &sp, PointArr &dp, CoefArr &cf) {
    sp = srcpos;
    dp = dstpos;
    cf = coef;
  };
};