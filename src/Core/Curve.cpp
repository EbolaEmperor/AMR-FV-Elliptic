#include "Curve.h"

#include "MPI.h"
#include "Tensor.h"
#include "Wrapper_LAPACKE.h"

#include <algorithm>
#include <limits>

template <int Dim, int Order>
int Curve<Dim, Order>::locatePiece(Real t) const {
  int np = polys.size();
  int left = 0, right = np - 1, mid;
  for (;;) {
    if (left == right)
      return left;
    mid = (left + right) / 2;
    if (t < knots[mid + 1])
      right = mid;
    else
      left = mid + 1;
  }
}

template <int Dim, int Order>
void Curve<Dim, Order>::concat(const T_Polynomial &p, Real plen) {
  if (knots.empty())
    knots.push_back(0.0);
  polys.push_back(p);
  knots.push_back(knots.back() + plen);
}

template <int Dim, int Order>
void Curve<Dim, Order>::concat(const Curve<Dim, Order> &pp) {
  if (pp.empty())
    return;
  if (knots.empty())
    knots.push_back(0.0);
  const auto &newknots = pp.getKnots();
  const auto &newpolys = pp.getPolys();
  Real delta = knots.back() - newknots.front();
  for (std::size_t i = 0; i < newpolys.size(); ++i) {
    polys.push_back(newpolys[i]);
    knots.push_back(newknots[i + 1] + delta);
  }
}

template <int Dim, int Order>
Curve<Dim, Order> Curve<Dim, Order>::reverse() const {
  Curve<Dim, Order> res;
  res.knots.resize(knots.size());
  res.polys.resize(polys.size());

  Real lastknot = knots.back();
  int j = polys.size() - 1;
  res.knots[0] = 0;
  for (std::size_t i = 0; i < polys.size(); i++, j--) {
    // flip the knots
    res.knots[i + 1] = lastknot - knots[j];
    // flip the parametrization
    res.polys[i] = polys[j].translate(knots[j + 1] - knots[j],
                                      true);  // reflect = true
  }
  return res;
}

template <int Dim, int Order>
Curve<Dim, Order> Curve<Dim, Order>::extract(Real lo,
                                             Real hi,
                                             Real tol) const {
  if (hi <= lo + tol)
    return Curve<Dim, Order>();

  int ihead = locatePiece(lo);
  int itail = locatePiece(hi);
  // avoid pieces with tiny length
  if (lo >= knots[ihead + 1] - tol)
    ihead++;
  if (hi <= knots[itail] + tol)
    itail--;
  assert(ihead <= itail);

  Curve<Dim, Order> res;
  res.knots.resize(itail - ihead + 2);
  res.polys.resize(itail - ihead + 1);
  res.knots[0] = lo;
  res.polys[0] = polys[ihead].translate(lo - knots[ihead]);

  // range [ihead+1, itail]
  std::copy(&knots[ihead + 1], &knots[itail + 1], &(res.knots[1]));
  std::copy(&polys[ihead + 1], &polys[itail + 1], &(res.polys[1]));
  // mark the tail
  res.knots.back() = hi;

  return res;
}

template <int Dim, int Order>
inline void Curve<Dim, Order>::split(const vector<Real> &brks,
                                     vector<Curve<Dim, Order>> &out,
                                     Real tol) const {
  bool closed = isClosed(tol);
  if (brks.empty()) {
    out.push_back(*this);
    return;
  }
  auto head = extract(knots.front(), brks.front(), tol);
  if (!closed && !head.empty())
    out.push_back(std::move(head));
  std::size_t i = 0;
  for (; i < brks.size() - 1; ++i) {
    if (brks[i + 1] > brks[i] + tol)
      out.push_back(extract(brks[i], brks[i + 1], tol));
  }
  auto tail = extract(brks[i], knots.back(), tol);
  if (!closed) {
    if (!tail.empty())
      out.push_back(std::move(tail));
  } else {
    tail.concat(head);
    if (!tail.empty())
      out.push_back(std::move(tail));
  }
}

template <int Dim, int Order>
Curve<Dim, Order> Curve<Dim, Order>::makeMonotonic(Real tol) const {
  Curve<Dim, Order> res;
  int np = knots.size() - 1;
  for (int i = 0; i < np; i++) {
    std::vector<Real> ex;
    // find out all the monotonic pieces by first locating the extrema
    for (int d = 0; d < Dim; d++) {
      auto rp = getComp(polys[i], d);
      extrema(rp, std::back_inserter(ex), tol);
    }
    // filter out the extrema out of domain
    // note that polys[i] is expressed in the variable (t-knots[i])
    Interval<1> validDomain(Vec<Real, 1>(0.0),
                            Vec<Real, 1>(knots[i + 1] - knots[i]));
    auto tbr1 =
        std::remove_if(ex.begin(), ex.end(), [&](const Real &a) -> bool {
          return !validDomain.contain(a, (Real)0);
        });
    // zero-tolerance is fine ^
    std::sort(ex.begin(), tbr1);
    ex.erase(std::unique(
                 ex.begin(),
                 tbr1,
                 [&tol](const Real &a, const Real &b) { return b - a < tol; }),
             ex.end());

    if (!ex.empty()) {
      // mark the head & tail while avoiding tiny pieces
      if (ex.front() > tol)
        ex.insert(ex.begin(), (Real)0);
      else
        ex.front() = 0;
      if (ex.back() < validDomain.hi()[0] - tol)
        ex.push_back(validDomain.hi()[0]);
      else
        ex.back() = validDomain.hi()[0];
      for (std::size_t j = 0; j < ex.size() - 1; j++)
        res.concat(polys[i].translate(ex[j]), ex[j + 1] - ex[j]);
    } else {
      // just copy if it is already mono
      res.concat(polys[i], knots[i + 1] - knots[i]);
    }
  }
  return res;
}

template <int Dim, int Order>
int Curve<Dim, Order>::countProperInts(Real c, int d, Real tol) const {
  int res = 0;
  auto process = [&c, &d, &tol](const rVec &p0, const rVec &p1) -> int {
    Real lo = std::min(p0[d], p1[d]);
    Real hi = std::max(p0[d], p1[d]);
    if (hi - lo <= tol)  // parallel to axis
      return 0;
    if (hi < c - tol || lo > c + tol)  // disjoint case
      return 0;
    if (hi <= c + tol || lo >= c - tol) {  // improper case
      auto mid = (p0 + p1) * 0.5;
      return (mid[d] > c) ? (1) : (0);
    }
    return 1;
  };
  for (std::size_t i = 0; i < polys.size(); ++i)
    res += process(polys[i][0], polys[i](knots[i + 1] - knots[i]));
  return res;
}

//============================================================

template <int Dim, int Order>
void Curve<Dim, Order>::dump(std::ostream &os) const {
  int header[] = {Dim, Order};
  os.write((char *)header, sizeof(header));

  int np = polys.size();
  os.write((char *)&np, sizeof(np));

  os.write((char *)knots.data(), sizeof(Real) * (np + 1));
  for (int i = 0; i < np; i++) {
    for (int j = 0; j < Dim; j++) {
      auto rp = getComp(polys[i], j);
      os.write((char *)(&rp[0]), sizeof(Real) * Order);
    }
  }
}

template <int Dim, int Order>
Curve<Dim, Order> Curve<Dim, Order>::load(std::istream &is) {
  Curve<Dim, Order> res;
  std::vector<Real> &knots = res.knots;
  std::vector<Polynomial<Order, rVec>> &polys = res.polys;

  int tmp[2], np;
  is.read((char *)tmp, sizeof(int) * 2);
  assert(tmp[0] == Dim && tmp[1] == Order);
  is.read((char *)&np, sizeof(int));
  knots.resize(np + 1);
  polys.resize(np);
  is.read((char *)knots.data(), sizeof(Real) * (np + 1));
  for (int i = 0; i < np; i++) {
    Polynomial<Order, rVec> p;
    Real buf[Dim * Order];
    Real *pbuf = buf;
    is.read((char *)buf, sizeof(Real) * Dim * Order);
    for (int j = 0; j < Dim; j++)
      for (int k = 0; k < Order; k++)
        p[k][j] = *pbuf++;
    polys[i] = p;
  }
  return res;
}

//============================================================

template <int Order>
Curve<2, Order> createLineSegment(const Vec<Real, 2> &p0,
                                  const Vec<Real, 2> &p1) {
  Curve<2, Order> res;
  Real l = norm(p1 - p0, 2);
  res.knots = {0.0, l};
  res.polys.resize(1);
  res.polys.front()[0] = p0;
  res.polys.front()[1] = (p1 - p0) / l;
  return res;
};

template <int Order>
Curve<2, Order> createRect(const Vec<Real, 2> &lo, const Vec<Real, 2> &hi) {
  Curve<2, Order> res;
  auto &knots = res.knots;
  auto &polys = res.polys;
  res.knots.resize(5);
  res.polys.resize(4);
  knots[0] = 0.0;
  knots[1] = hi[0] - lo[0];
  knots[2] = knots[1] + hi[1] - lo[1];
  knots[3] = knots[2] + knots[1];
  knots[4] = 2 * knots[2];
  polys[0][0] = lo;
  polys[0][1] = {1.0, 0.0};
  polys[1][0] = {hi[0], lo[1]};
  polys[1][1] = {0.0, 1.0};
  polys[2][0] = hi;
  polys[2][1] = {-1.0, 0.0};
  polys[3][0] = {lo[0], hi[1]};
  polys[3][1] = {0.0, -1.0};
  return res;
}

template <int Order>
Real area(const Curve<2, Order> &gon) {
  Real a = 0.0;
  const auto &knots = gon.getKnots();
  int i = 0;
  // apply the Green's formula
  for (const auto &p : gon.getPolys()) {
    auto x = getComp(p, 0);
    auto y = getComp(p, 1);
    auto dy = y.der();
    a += (x * dy).prim()(knots[i + 1] - knots[i]);
    ++i;
  }
  return a;
}

template <int Order>
Real arclength(const Curve<2, Order> &c) {
  Real l = 0.0;
  const auto &knots = c.getKnots();
  const auto &polys = c.getPolys();
  for (std::size_t k = 0; k < polys.size(); ++k) {
    auto dt = knots[k + 1] - knots[k];
    auto dxdt = getComp(polys[k], 0).der();
    auto dydt = getComp(polys[k], 1).der();
    auto ndsdt = [&](Real t) {
      Vec<Real, 2> ds{dxdt(t), dydt(t)};
      return norm(ds, 2);
    };
    l += aquad(ndsdt, 0, dt);
  }
  return l;
}

template <int Dim, int Order>
Interval<Dim> boundingBox(const Curve<Dim, Order> &c) {
  std::vector<Curve<Dim, Order>> vc{c};
  return boundingBox(vc);
}

template <int Dim, int Order>
Interval<Dim> boundingBox(const std::vector<Curve<Dim, Order>> &vc) {
  Vec<Real, Dim> lower(std::numeric_limits<Real>::max());
  Vec<Real, Dim> upper = -lower;
  for (const auto &c : vc) {
    const auto &polys = c.getPolys();
    for (std::size_t i = 0; i < polys.size(); ++i) {
      lower = min(lower, polys[i][0]);
      upper = max(upper, polys[i][0]);
    }
    lower = min(lower, c.endpoint());
    upper = max(upper, c.endpoint());
  }
  return Interval<Dim>(lower, upper);
}

template <int Dim, int Order>
Curve<Dim, Order - 1> der(const Curve<Dim, Order> &c) {
  Curve<Dim, Order - 1> res;
  res.knots = c.getKnots();
  for (const auto &pn : c.getPolys())
    res.polys.push_back(pn.der());
  return res;
}

//=================================================

template <>
Curve<2, 2> fitCurve(const std::vector<Vec<Real, 2>> &knots, bool) {
  const int Order = 2;
  auto numKnots = knots.size();
  assert(numKnots >= 2);
  Curve<2, Order> res;
  for (std::size_t i = 0; i < numKnots - 1; ++i) {
    Polynomial<2, Vec<Real, 2>> p;
    Real l = norm(knots[i + 1] - knots[i], 2);
    p[0] = knots[i];
    p[1] = (knots[i + 1] - knots[i]) / l;
    res.concat(p, l);
  }
  return res;
}

template <>
Curve<2, 4> fitCurve(const std::vector<Vec<Real, 2>> &vertices,
                     bool periodic) {
  assert(periodic);
  using rVec = Vec<Real, 2>;
  const int Order = 4;
  const int numPiece = vertices.size() - 1;
  int k;
  // calculate the accumulated chordal length
  Tensor<Real, 1> t(numPiece + 1);
  t(0) = 0.0;
  for (k = 1; k <= numPiece; ++k)
    t(k) = t(k - 1) + norm(vertices[k] - vertices[k - 1], 2);
  // prepare the coefficient matrix
  Tensor<Real, 2> mat(numPiece);
  mat = 0.0;
  mat(0, 0) = 2.0;
  mat(0, 1) = t(1) / (t(1) + t(numPiece) - t(numPiece - 1));
  mat(0, numPiece - 1) =
      (t(numPiece) - t(numPiece - 1)) / (t(1) + t(numPiece) - t(numPiece - 1));
  for (k = 1; k < numPiece - 1; ++k) {
    mat(k, k - 1) = (t(k) - t(k - 1)) / (t(k + 1) - t(k - 1));
    mat(k, k) = 2.0;
    mat(k, k + 1) = (t(k + 1) - t(k)) / (t(k + 1) - t(k - 1));
  }
  mat(k, k - 1) = (t(k) - t(k - 1)) / (t(k + 1) - t(k - 1));
  mat(k, k) = 2.0;
  mat(k, 0) = (t(k + 1) - t(k)) / (t(k + 1) - t(k - 1));
  // prepare the RHS
  Tensor<Real, 2> rhs(Vec<int, 2>{numPiece, 2});
  const auto &verts = vertices;
  for (int k = 1; k <= numPiece - 1; ++k) {
    rhs(k, 0) = (verts[k + 1][0] - verts[k][0]) / (t(k + 1) - t(k)) -
                (verts[k][0] - verts[k - 1][0]) / (t(k) - t(k - 1));
    rhs(k, 0) *= 6.0 / (t(k + 1) - t(k - 1));
    rhs(k, 1) = (verts[k + 1][1] - verts[k][1]) / (t(k + 1) - t(k)) -
                (verts[k][1] - verts[k - 1][1]) / (t(k) - t(k - 1));
    rhs(k, 1) *= 6.0 / (t(k + 1) - t(k - 1));
  }
  rhs(0, 0) =
      (verts[1][0] - verts[0][0]) / t(1) -
      (verts[0][0] - verts[numPiece - 1][0]) / (t(numPiece) - t(numPiece - 1));
  rhs(0, 0) *= 6.0 / (t(1) + t(numPiece) - t(numPiece - 1));
  rhs(0, 1) =
      (verts[1][1] - verts[0][1]) / t(1) -
      (verts[0][1] - verts[numPiece - 1][1]) / (t(numPiece) - t(numPiece - 1));
  rhs(0, 1) *= 6.0 / (t(1) + t(numPiece) - t(numPiece - 1));
  // solve the linear system (not a tri-diagonal system under periodic
  // condition)
  Tensor<int, 1> ipiv(numPiece);
  auto info = LAPACKE_dgesv(LAPACK_COL_MAJOR,
                            numPiece,
                            2,
                            mat.data(),
                            numPiece,
                            ipiv.data(),
                            rhs.data(),
                            numPiece);
  if (info != 0)
    throw std::runtime_error("fitCurve() - DGESV");
  // assemble the spline
  Curve<2, Order> res;
  res.knots = std::vector<Real>(t.data(), t.data() + numPiece + 1);
  res.polys.resize(numPiece);
  for (k = 0; k < numPiece - 1; ++k) {
    res.polys[k][0] = rVec{verts[k][0], verts[k][1]};
    res.polys[k][1] = (verts[k + 1] - verts[k]) / (t(k + 1) - t(k));
    res.polys[k][1] = res.polys[k][1] - rVec{rhs(k + 1, 0) + 2 * rhs(k, 0),
                                             rhs(k + 1, 1) + 2 * rhs(k, 1)} *
                                            (t(k + 1) - t(k)) / 6.0;
    res.polys[k][2] = rVec{rhs(k, 0), rhs(k, 1)} / 2.0;
    res.polys[k][3] =
        rVec{rhs(k + 1, 0) - rhs(k, 0), rhs(k + 1, 1) - rhs(k, 1)} /
        (t(k + 1) - t(k)) / 6.0;
  }
  res.polys[k][0] = rVec{verts[k][0], verts[k][1]};
  res.polys[k][1] = (verts[k + 1] - verts[k]) / (t(k + 1) - t(k));
  res.polys[k][1] = res.polys[k][1] - rVec{rhs(0, 0) + 2 * rhs(k, 0),
                                           rhs(0, 1) + 2 * rhs(k, 1)} *
                                          (t(k + 1) - t(k)) / 6.0;
  res.polys[k][2] = rVec{rhs(k, 0), rhs(k, 1)} / 2.0;
  res.polys[k][3] = rVec{rhs(0, 0) - rhs(k, 0), rhs(0, 1) - rhs(k, 1)} /
                    (t(k + 1) - t(k)) / 6.0;
  return res;
}

template <int Dim, int Order>
inline size_t Curve<Dim, Order>::linearSize() const {
  size_t size = 0;
  size += sizeof(int);
  size += sizeof(Real) * polys.size();                // knots
  size += sizeof(Real) * polys.size() * Order * Dim;  // polys.
  return size;
};

template <int Dim, int Order>
inline void Curve<Dim, Order>::linearIn(void *buf,
                                        const int bufSize,
                                        int &bufPos) {
  knots.clear();
  polys.clear();
  std::vector<Real> knotsInfo;
  std::vector<Real> polysInfo;
  int numPieces;
  MPI_Unpack(buf, bufSize, &bufPos, &numPieces, 1, MPI_INT, MPI_COMM_WORLD);
  // std::cout << numPieces << "\n";
  knotsInfo.resize(numPieces);
  polysInfo.resize(numPieces * Order * Dim);
  MPI_Unpack(buf,
             bufSize,
             &bufPos,
             knotsInfo.data(),
             knotsInfo.size(),
             MPI_DOUBLE,
             MPI_COMM_WORLD);
  MPI_Unpack(buf,
             bufSize,
             &bufPos,
             polysInfo.data(),
             polysInfo.size(),
             MPI_DOUBLE,
             MPI_COMM_WORLD);
  int cnt = 0;
  for (int i = 0; i != numPieces; ++i) {
    Polynomial<Order, rVec> p;
    for (int j = 0; j != Order; j++) {
      for (int d = 0; d != Dim; d++) {
        p[j][d] = polysInfo[cnt++];
      }
    }
    this->concat(p, knotsInfo[i]);
  }
};

template <int Dim, int Order>
inline void Curve<Dim, Order>::linearOut(void *buf,
                                         const int bufSize,
                                         int &bufPos) const {
  std::vector<Real> knotsInfo;
  std::vector<Real> polysInfo;
  int numPieces = polys.size();
  for (size_t i = 0; i != polys.size(); ++i) {
    knotsInfo.push_back(knots[i + 1] - knots[i]);
    for (int j = 0; j != Order; ++j) {
      const rVec &coef = polys[i][j];
      for (int d = 0; d != Dim; ++d) {
        polysInfo.push_back(coef[d]);
      }
    }
  }
  MPI_Pack(&numPieces, 1, MPI_INT, buf, bufSize, &bufPos, MPI_COMM_WORLD);
  MPI_Pack(knotsInfo.data(),
           knotsInfo.size(),
           MPI_DOUBLE,
           buf,
           bufSize,
           &bufPos,
           MPI_COMM_WORLD);
  MPI_Pack(polysInfo.data(),
           polysInfo.size(),
           MPI_DOUBLE,
           buf,
           bufSize,
           &bufPos,
           MPI_COMM_WORLD);
};

//=================================================
// explicit instantiation of the followings

template class Curve<2, 1>;
template class Curve<2, 2>;
template class Curve<2, 3>;
template class Curve<2, 4>;

template Curve<2, 3> der(const Curve<2, 4> &);
template Curve<2, 2> der(const Curve<2, 3> &);
template Curve<2, 1> der(const Curve<2, 2> &);

template Interval<2> boundingBox(const Curve<2, 2> &);
template Interval<2> boundingBox(const Curve<2, 4> &);

template Real area(const Curve<2, 2> &gon);
template Real area(const Curve<2, 4> &gon);

template Real arclength(const Curve<2, 2> &);
template Real arclength(const Curve<2, 4> &);

template Curve<2, 2> createRect(const Vec<Real, 2> &, const Vec<Real, 2> &);
template Curve<2, 4> createRect(const Vec<Real, 2> &, const Vec<Real, 2> &);

template Curve<2, 2> createLineSegment(const Vec<Real, 2> &p0,
                                       const Vec<Real, 2> &p1);
template Curve<2, 4> createLineSegment(const Vec<Real, 2> &p0,
                                       const Vec<Real, 2> &p1);
