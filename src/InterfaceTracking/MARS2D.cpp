#include "MARS2D.h"

#include "Core/Polynomial.h"
#include "Core/Tensor.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <list>
#include <sstream>
#include <type_traits>

Real tol = 1e-15;

using namespace std;

using Point = Vec<Real, 2>;

using Crv = Curve<2, 4>;

template <class T>
using Vector = vector<T>;

// VectorFunction version

template <int Order>
void MARS2D<Order, VectorFunction>::discreteFlowMap(const VectorFunction<2> &v,
                                                    Vector<Point> &pts,
                                                    Real tn,
                                                    Real dt) {
  Base::TI->timeStep(v, pts, tn, dt);
  return;
}

/**
 * @brief helper function of removeSmallEdges
 * @tparam Container
 * @param  ids              id of the current points
 * @param  pts
 * @param  lowBound
 * @return has removed points or not
 */
bool removeMarkers(Vector<unsigned int> &ids,
                   Vector<Point> &pts,
                   Real lowBound) {
  // set distances
  int num = ids.size();
  if (num == 2)
    return false;

  Vector<Real> dist(num - 1);
  auto pit = pts.begin();
  Point prept = *pit;
  for (int i = 0; i < num - 1; i++) {
    ++pit;
    dist[i] = norm(*pit - prept);
    prept = *pit;
  }

  bool predelete = false;  // mark whether the pre-point has been erased
  auto it = ids.begin();
  pit = pts.begin();
  ++it;
  ++pit;

  int i = 1;
  if (dist[0] < lowBound) {
    predelete = true;
    it = ids.erase(it);
    pit = pts.erase(pit);
    i++;
  }

  while (i < num - 2) {
    if (dist[i] >= lowBound) {
      predelete = false;
      ++it;
      ++pit;
      i++;
    } else {
      if (dist[i - 1] <= dist[i + 1]) {
        if (predelete == false) {
          it = ids.erase(it);
          pit = pts.erase(pit);
          i++;
          predelete = true;
        } else {
          ++it;
          ++pit;
          i++;
          predelete = false;
        }
      } else {
        it = ids.erase(++it);
        pit = pts.erase(++pit);
        i += 2;
        predelete = true;
      }
    }
  }
  if (i == num - 1) {
    return num != (int)ids.size();
  } else {
    if (predelete == true || dist[i] >= lowBound)
      return num != (int)ids.size();
    it = ids.erase(it);
    pit = pts.erase(pit);
    return true;
  }
}

template <int Order>
Vector<unsigned int> MARS2D<Order, VectorFunction>::removeSmallEdges(
    Vector<Point> &pts) {
  int num = pts.size();
  Vector<unsigned int> ids(num);
  for (int i = 0; i < num; i++) {
    ids[i] = i;
  }

  while (true) {
    if (!removeMarkers(ids, pts, (chdLenRange.lo())[0]))
      break;
  }

  Vector<unsigned int> rids(num - ids.size());
  int it = 0;
  unsigned int allid = 0;
  int count = 0;
  while (count < (int)rids.size() - 1) {
    if (ids[it] != allid) {
      rids[count] = allid;
      count++;
      allid++;
    } else {
      it++;
      allid++;
    }
  }
  return rids;
}

template <int Order>
Vector<unsigned int> MARS2D<Order, VectorFunction>::splitLongEdges(
    const VectorFunction<2> &v,
    Vector<Point> &pts,
    const Crv &crv,
    Real tn,
    Real dt) {
  assert(crv.isClosed(tol));

  int num = pts.size();
  Vector<unsigned int> ids;
  Vector<Polynomial<Order, Point>> polys = crv.getPolys();
  Polynomial<Order, Point> lpoly;
  Vector<Real> knots = crv.getKnots();
  function<void(Real, Real, Vector<Real> &)> split;

  // lambda: split between two points
  split = [&](Real left, Real right, Vector<Real> &chdt) -> void {
    Point lpt = lpoly(left);
    lpt = Base::TI->timeStep(v, lpt, tn, dt);
    Point rpt = lpoly(right);
    rpt = Base::TI->timeStep(v, rpt, tn, dt);
    if (norm(rpt - lpt) > (chdLenRange.hi())[0]) {
      int N = (int)ceil(norm(rpt - lpt) / (chdLenRange.hi())[0]);
      Real dl = (right - left) / N;
      for (int j = 0; j < N - 1; j++) {
        split(left + dl * j, left + dl * (j + 1), chdt);
        chdt.push_back(left + dl * (j + 1));
      }
      split(right - dl, right, chdt);
    } else {
      return;
    }
  };

  auto it = pts.begin();
  unsigned int count = 0;
  Point prept = *it;
  ++it;
  Real dist;

  for (int i = 0; i < num - 1; i++) {
    dist = norm(*it - prept);
    if (dist <= (chdLenRange.hi())[0]) {
      prept = *it;
      ++it;
      count++;
      continue;
    }

    prept = *it;
    lpoly = polys[i];
    Vector<Real> chdt;  // save parameter t of the added points
    split(0, knots[i + 1] - knots[i], chdt);

    for (int j = chdt.size() - 1; j >= 0; j--) {
      count++;
      ids.push_back(count);
      Point opt = lpoly(chdt[j]);
      it = pts.emplace(it, Base::TI->timeStep(v, opt, tn, dt));
    }
    count++;
    std::advance(it, (int)chdt.size() + 1);
  }
  return ids;
}

template <int Order>
void MARS2D<Order, VectorFunction>::timeStep(const VectorFunction<2> &v,
                                             YS &ys,
                                             Real tn,
                                             Real dt) {
  Vector<Crv> vcrv = ys.getBoundaryCycles();
  int id = 1;
  for (auto &crv : vcrv) {
    assert(crv.isClosed(tol));

    // pts: now's points
    Vector<Polynomial<Order, Point>> polys = crv.getPolys();
    Vector<Point> pts(polys.size() + 1);
    for (int i = 0; i < (int)polys.size(); i++) {
      pts[i] = polys[i][0];
    }
    pts[(int)polys.size()] = polys[0][0];

    // get the points after discrete flow map
    discreteFlowMap(v, pts, tn, dt);

    // split long edges
    Vector<unsigned int> splitids = splitLongEdges(v, pts, crv, tn, dt);

    // remove small edges
    Vector<unsigned int> removeids = removeSmallEdges(pts);

    //
    int num = pts.size();
    Vector<Real> dist(num - 1);

    // use Container<Point> pts to generate the Vector<Point> pts
    // fitCurve only support Vector<Point> version
    for (int i = 0; i < num - 1; i++) {
      dist[i] = norm(pts[i + 1] - pts[i], 2);
    }
    crv = fitCurve<Order>(pts, periodic);

    auto maxp = max_element(dist.begin(), dist.end());
    auto minp = min_element(dist.begin(), dist.end());

    cout << "Curve " << id << ":  Add " << splitids.size() << " Points,"
         << " remove " << removeids.size() << " Points."
         << " Max chdlength: " << *maxp << " Min chdlength: " << *minp << endl;
    id++;
  }
  ys = YS(SegmentedRealizableSpadjor<Order>(vcrv), tol);
  return;
}

// VectorOnHypersurface version, with unique difference in
// splitLongEdges.

template <int Order>
void MARS2D<Order, VectorOnHypersurface>::discreteFlowMap(
    const VectorOnHypersurface<2> &v,
    Vector<Point> &pts,
    Real tn,
    Real dt) {
  Base::TI->timeStep(v, pts, tn, dt);
  return;
}

template <int Order>
Vector<unsigned int> MARS2D<Order, VectorOnHypersurface>::removeSmallEdges(
    Vector<Point> &pts) {
  int num = pts.size();
  Vector<unsigned int> ids(num);
  for (int i = 0; i < num; i++) {
    ids[i] = i;
  }

  while (true) {
    if (!removeMarkers(ids, pts, (chdLenRange.lo())[0]))
      break;
  }

  Vector<unsigned int> rids(num - ids.size());
  int it = 0;
  unsigned int allid = 0;
  int count = 0;
  while (count < (int)rids.size() - 1) {
    if (ids[it] != allid) {
      rids[count] = allid;
      count++;
      allid++;
    } else {
      it++;
      allid++;
    }
  }
  return rids;
}

template <int Order>
bool splitMarkers(Vector<bool> &ids,
                  Vector<Point> &oldpts,
                  const Curve<2, Order> &crv,
                  const Vector<Real> &dist,
                  Real highBound) {
  int num = oldpts.size();
  Vector<Polynomial<Order, Point>> polys = crv.getPolys();
  Polynomial<Order, Point> lpoly;
  Vector<Real> knots = crv.getKnots();

  auto it = oldpts.begin();
  auto idit = ids.begin();
  ++it;
  ++idit;
  Point opt;
  int n;
  Real dt;

  for (int i = 0; i < num - 1; i++) {
    if (dist[i] <= highBound) {
      ++it;
      ++idit;
      continue;
    }

    lpoly = polys[i];
    n = ceil(dist[i] / highBound);
    dt = (knots[i + 1] - knots[i]) / n;

    for (int j = n - 1; j >= 1; j--) {
      opt = lpoly(dt * j);
      it = oldpts.emplace(it, opt);
      idit = ids.emplace(idit, true);
    }
    std::advance(it, n);
    std::advance(idit, n);
  }
  return num != (int)oldpts.size();
}

template <int Order>
Vector<unsigned int> MARS2D<Order, VectorOnHypersurface>::splitLongEdges(
    const VectorOnHypersurface<2> &v,
    Vector<Point> &pts,
    const Crv &crv,
    Real tn,
    Real dt) {
  int num = pts.size();
  auto polys = crv.getPolys();
  Vector<Point> oldpts(num);

  for (int i = 0; i < num - 1; i++) {
    oldpts[i] = polys[i][0];
  }
  oldpts[num - 1] = polys[0][0];

  Vector<bool> ids(num, false);
  Vector<Real> dist(num - 1);

  while (true) {
    for (int i = 0; i < (int)dist.size(); i++) {
      dist[i] = norm(pts[i + 1] - pts[i], 2);
    }
    if (!splitMarkers(ids, oldpts, crv, dist, (chdLenRange.hi())[0]))
      break;
    pts = oldpts;
    Base::TI->timeStep(v, pts, tn, dt);
    dist.resize(pts.size() - 1);
  }

  Vector<unsigned int> aids(ids.size() - num);
  int count = 0;
  for (int i = 0; i < (int)ids.size(); i++) {
    if (ids[i] == true) {
      aids[count] = i;
      count++;
    }
  }
  return aids;
}

template <int Order>
void MARS2D<Order, VectorOnHypersurface>::timeStep(
    const VectorOnHypersurface<2> &v,
    YS &ys,
    Real tn,
    Real dt) {
  Vector<Crv> vcrv = ys.getBoundaryCycles();
  int id = 1;
  for (auto &crv : vcrv) {
    assert(crv.isClosed(tol));

    // pts: now's points
    Vector<Polynomial<Order, Point>> polys = crv.getPolys();
    Vector<Point> pts(polys.size() + 1);
    for (int i = 0; i < (int)polys.size(); i++) {
      pts[i] = polys[i][0];
    }
    pts[(int)polys.size()] = polys[0][0];

    // get the points after discrete flow map
    discreteFlowMap(v, pts, tn, dt);

    // split long edges
    Vector<unsigned int> splitids = splitLongEdges(v, pts, crv, tn, dt);

    // remove small edges
    Vector<unsigned int> removeids = removeSmallEdges(pts);

    //
    int num = pts.size();
    Vector<Real> dist(num - 1);

    // use Container<Point> pts to generate the Vector<Point> pts
    // fitCurve only support Vector<Point> version
    for (int i = 0; i < num - 1; i++) {
      dist[i] = norm(pts[i + 1] - pts[i], 2);
    }
    crv = fitCurve<Order>(pts, periodic);

    auto maxp = max_element(dist.begin(), dist.end());
    auto minp = min_element(dist.begin(), dist.end());

    cout << "Curve " << id << ":  Add " << splitids.size() << " Points,"
         << " remove " << removeids.size() << " Points."
         << " Max chdlength: " << *maxp << " Min chdlength: " << *minp << endl;
    id++;
  }
  ys = YS(SegmentedRealizableSpadjor<Order>(vcrv), tol);
  return;
}

template class MARS2D<2, VectorFunction>;
template class MARS2D<4, VectorFunction>;

template class MARS2D<2, VectorOnHypersurface>;
template class MARS2D<4, VectorOnHypersurface>;
