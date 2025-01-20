#ifndef POLYNOMIALSURFACE_H
#define POLYNOMIALSURFACE_H

#include "Core/BinaryPolynomial.h"
#include "YinSet3D/Plane.h"

#include <eigen3/Eigen/Eigen>
#include <vector>
// represent a binary polynomial.
template <int Order>
class PolynomialSurface {
private:
  YSB::Plane<Real> pla;                  // projecting plane
  BinaryPolynomial<Order, Real> bipoly;  // representation of the surface
public:
  PolynomialSurface() = default;
  explicit PolynomialSurface(const std::vector<YSB::Point<Real, 3>> &pts) {
    int n = pts.size();
    int coefNum = factorial(Order + 2) / (factorial(Order) * 2);
    assert(n >= coefNum && "Too less points!");
    pla = YSB::Plane<Real>(pts);
    int projectDim = pla.majorDim();
    std::vector<Vec<Real, 2>> projectedCoord;
    std::vector<Real> distToPlane;
    for (int i = 0; i < n; i++) {
      if (pla.contains(pts[i])) {
        projectedCoord.emplace_back(Vec<Real, 2>{
            pts[i][(projectDim + 1) % 3], pts[i][(projectDim + 2) % 3]});
        distToPlane.push_back(0.0);
      } else {
        YSB::Line<Real, 3> l(pts[i], pla.normalVec());
        YSB::Point<Real, 3> proP = pla.intersect(l);
        projectedCoord.emplace_back(Vec<Real, 2>{proP[(projectDim + 1) % 3],
                                                 proP[(projectDim + 2) % 3]});
        Real dist = (pts[i][projectDim] - proP[projectDim]) /
                    pla.normalVec()[projectDim];
        distToPlane.push_back(dist);
      }
    }
    Eigen::MatrixXd E(n, coefNum);
    Eigen::VectorXd b(n), X(n);
    for (int i = 0; i < n; i++) {
      int count = 0;
      for (int j = Order; j >= 0; j--) {
        for (int k = j; k >= 0; k--) {
          E(i, count++) =
              pow(projectedCoord[i][0], k) * pow(projectedCoord[i][1], j - k);
        }
      }
      b(i) = distToPlane[i];
    }
    X = E.colPivHouseholderQr().solve(b);
    std::vector<Real> coefs(coefNum);
    for (int i = 0; i < coefNum; i++)
      bipoly[i] = X(i);
  }

  explicit PolynomialSurface(const std::vector<YSB::Point<Real, 3>> &pts,
                             const YSB::Plane<Real> &apla) :
      pla(apla) {
    int n = pts.size();
    int coefNum = factorial(Order + 2) / (factorial(Order) * 2);
    assert(n >= coefNum && "Too less points!");
    int projectDim = pla.majorDim();
    std::vector<Vec<Real, 2>> projectedCoord;
    std::vector<Real> distToPlane;
    for (int i = 0; i < n; i++) {
      if (pla.contains(pts[i])) {
        projectedCoord.emplace_back(Vec<Real, 2>{
            pts[i][(projectDim + 1) % 3], pts[i][(projectDim + 2) % 3]});
        distToPlane.push_back(0.0);
      } else {
        YSB::Line<Real, 3> l(pts[i], pla.normalVec());
        YSB::Point<Real, 3> proP = pla.intersect(l);
        projectedCoord.emplace_back(Vec<Real, 2>{proP[(projectDim + 1) % 3],
                                                 proP[(projectDim + 2) % 3]});
        Real dist = (pts[i][projectDim] - proP[projectDim]) /
                    pla.normalVec()[projectDim];
        distToPlane.push_back(dist);
      }
    }
    Eigen::MatrixXd E(n, coefNum);
    Eigen::VectorXd b(n), X(n);
    for (int i = 0; i < n; i++) {
      int count = 0;
      for (int j = Order; j >= 0; j--) {
        for (int k = j; k >= 0; k--) {
          E(i, count++) =
              pow(projectedCoord[i][0], k) * pow(projectedCoord[i][1], j - k);
        }
      }
      b(i) = distToPlane[i];
    }
    X = E.colPivHouseholderQr().solve(b);
    std::vector<Real> coefs(coefNum);
    for (int i = 0; i < coefNum; i++)
      bipoly[i] = X(i);
  }
  const YSB::Plane<Real> &getPlane() const { return pla; }
  const BinaryPolynomial<Order, Real> &getBipoly() const { return bipoly; }
  Vec<Real, 3> getNormalVector(const YSB::Point<Real, 3> &p) const;
  YSB::Point<Real, 3> projectToSurface(const YSB::Point<Real, 3> &p) const;
};

template <int Order>
Vec<Real, 3> PolynomialSurface<Order>::getNormalVector(
    const YSB::Point<Real, 3> &p) const {
  int projectDim = pla.majorDim();
  Real u, v;
  if (pla.contains(p)) {
    u = p[(projectDim + 1) % 3];
    v = p[(projectDim + 2) % 3];
  } else {
    YSB::Line<Real, 3> l(p, pla.normalVec());
    auto proP = pla.intersect(l);
    u = proP[(projectDim + 1) % 3];
    v = proP[(projectDim + 2) % 3];
  }
  Vec<Real, 3> normalVec = pla.normalVec();
  Vec<Real, 3> du(0.0), dv(0.0);
  du[projectDim] = -normalVec[(projectDim + 1) % 3] / normalVec[projectDim];
  du[(projectDim + 1) % 3] = 1;
  du[(projectDim + 2) % 3] = 0;
  dv[projectDim] = -normalVec[(projectDim + 2) % 3] / normalVec[projectDim];
  dv[(projectDim + 1) % 3] = 0;
  dv[(projectDim + 2) % 3] = 1;

  Real dbipolyu = 0, dbipolyv = 0;
  for (int i = Order; i > 0; i--) {
    for (int j = i; j > 0; j--) {
      int count = (Order + i + 3) * (Order - i) / 2 + i - j;
      dbipolyu += bipoly[count] * j * pow(u, j - 1) * pow(v, i - j);
    }
  }
  for (int i = Order; i > 0; i--) {
    for (int j = i - 1; j >= 0; j--) {
      int count = (Order + i + 3) * (Order - i) / 2 + i - j;
      dbipolyv += bipoly[count] * (i - j) * pow(u, j) * pow(v, i - j - 1);
    }
  }
  du = du + normalVec * dbipolyu;
  dv = dv + normalVec * dbipolyv;
  Vec<Real, 3> res = normalize(cross(du, dv));
  return res;
}

template <int Order>
YSB::Point<Real, 3> PolynomialSurface<Order>::projectToSurface(
    const YSB::Point<Real, 3> &p) const {
  YSB::Point<Real, 3> res;
  int projectDim = pla.majorDim();
  if (pla.contains(p)) {
    Real dist = bipoly(p[(projectDim + 1) % 3], p[(projectDim + 2) % 3]);
    res = p + pla.normalVec() * dist;
  } else {
    YSB::Line<Real, 3> l(p, pla.normalVec());
    auto proP = pla.intersect(l);
    Real dist = bipoly(proP[(projectDim + 1) % 3], proP[(projectDim + 2) % 3]);
    res = proP + pla.normalVec() * dist;
  }
  return res;
}
#endif