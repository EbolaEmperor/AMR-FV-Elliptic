#include "MARS2DIMV.h"

#include "Core/Polynomial.h"
#include "YinSet/SimplicialComplex.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <list>
#include <sstream>
#include <type_traits>
#include <utility>

using namespace std;

using Point = Vec<Real, 2>;

template <class T>
using Vector = vector<T>;

static Real tol = 1e-12;

// VectorFunction version

template <int Order>
void MARS2DIMV<Order, VectorFunction>::discreteFlowMap(
    const VectorFunction<2> &v,
    Vector<Point> &pts,
    Real tn,
    Real dt) {
  Base::TI->timeStep(v, pts, tn, dt);
  return;
}

/**
 * @brief remove cycle: do not remove the first and the last point, do not
 * remove points consecutively
 * @tparam Container
 * @param  ids              id of the current points
 * @param  pts
 * @param  lowBound
 * @return has removed points or not
 */
bool removeIMV(Vector<unsigned int> &ids, Vector<Point> &pts, Real lowBound) {
  // set distances
  int num = pts.size();
  if (num == 2)
    return false;

  int count = 1;
  int i = 1;

  // function<void(int&, int&)> addpi;

  // lambda: split between two points
  auto addpi = [&]() -> void {
    if (count != i) {
      pts[count] = pts[i];
      ids[count] = ids[i];
    }
    count++;
    i++;
    return;
  };

  Vector<Real> dist(num - 1);
  for (int j = 0; j < num - 1; j++) {
    dist[j] = norm(pts[j + 1] - pts[j], 2);
  }

  bool predelete = false;  // mark whether the pre-point has been erased
  if (dist[0] < lowBound) {
    predelete = true;
    i++;
  }

  while (i < num - 2) {
    if (dist[i] >= lowBound) {
      predelete = false;
      addpi();
    } else {
      if (dist[i - 1] <= dist[i + 1]) {
        if (predelete == false) {
          i++;
          predelete = true;
        } else {
          addpi();
          predelete = false;
        }
      } else {
        addpi();
        i++;
        predelete = true;
      }
    }
  }
  if (i == num - 1) {
    addpi();
    pts.resize(count);
    ids.resize(count);
    return num != count;
  } else {
    if (predelete == true || dist[i] >= lowBound) {
      addpi();
      addpi();
      pts.resize(count);
      ids.resize(count);
      return num != count;
    }
    i++;
    addpi();
    pts.resize(count);
    ids.resize(count);
    return true;
  }
}

template <int Order>
Vector<unsigned int> MARS2DIMV<Order, VectorFunction>::removeSmallEdges(
    Vector<Point> &pts) {
  int num = pts.size();
  Vector<unsigned int> ids(num);
  for (int i = 0; i < num; i++) {
    ids[i] = i;
  }

  while (true) {
    // output ids, debug use
    /*
    std::cout << "(";
    for (auto &id : ids)
    {
        std::cout << id << ", ";
    }
    std::cout << "\b)" << std::endl;
    */

    if (!removeIMV(ids, pts, (chdLenRange.lo())[0]))
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
Vector<unsigned int> MARS2DIMV<Order, VectorFunction>::splitLongEdges(
    const VectorFunction<2> &v,
    Vector<Point> &pts,
    const Vector<Real> &knots,
    const Vector<Polynomial<Order, Point>> &polys,
    Real tn,
    Real dt) {
  // assert(crv.isClosed(tol));

  int num = pts.size();
  Vector<Point> res(2 * num);
  Vector<unsigned int> ids;
  Polynomial<Order, Point> lpoly;
  function<void(Real, Real, Vector<Real> &)> split;

  // lambda: split between two points
  split = [&](Real left, Real right, Vector<Real> &chdt) {
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
    }
    return;
  };

  auto addpt = [&](int &count, Point pt) {
    res[count] = pt;
    count++;
    if (count >= (int)res.size())
      res.resize(2 * count);
    return;
  };

  unsigned int id = 0;
  int count = 0;
  addpt(count, pts[0]);
  Real dist;
  Point opt;

  for (int i = 0; i < num - 1; i++) {
    dist = norm(pts[i + 1] - pts[i]);
    if (dist <= (chdLenRange.hi())[0]) {
      addpt(count, pts[i + 1]);
      id++;
      continue;
    }

    lpoly = polys[i];
    Vector<Real> chdt;  // save parameter t of the added points
    split(0, knots[i + 1] - knots[i], chdt);

    for (int j = 0; j < (int)chdt.size(); j++) {
      id++;
      ids.push_back(id);
      opt = lpoly(chdt[j]);
      addpt(count, Base::TI->timeStep(v, opt, tn, dt));
    }
    id++;
    addpt(count, pts[i + 1]);
  }
  res.resize(count);
  pts = res;
  return ids;
}

template <int Order>
void MARS2DIMV<Order, VectorFunction>::timeStep(const VectorFunction<2> &v,
                                                YS &ys,
                                                Real tn,
                                                Real dt) {
  Vector<Crv> vcrv = ys.getBoundaryCycles();
  int N = vcrv.size();
  // get kinks index
  SimplicialComplex kinks = ys.getKinks();
  Vector<Vector<unsigned int>> vbrk(N);
  if (kinks.getSimplexes().size() != 0) {
    for (auto &simplex : kinks.getSimplexes()[0]) {
      unsigned int index = *simplex.vertices.begin();
      std::pair<unsigned int, unsigned int> id;
      int info = ys.vertex2Point(index, id);
      if (info == 0)
        throw std::runtime_error("cannot find kink's id");
      vbrk[id.first].push_back(id.second);
    }
  }

  Vector<std::pair<unsigned int, unsigned int>> newkinks;

  for (int id = 0; id < N; id++) {
    Crv &crv = vcrv[id];
    assert(crv.isClosed(tol));
    // get polys and knots
    const Vector<Polynomial<Order, Point>> &polys = crv.getPolys();
    const Vector<Real> &knots = crv.getKnots();
    unsigned int n = knots.size();
    // get present pts
    Vector<Point> pts(n);
    for (unsigned int i = 0; i < n - 1; i++) {
      pts[i] = polys[i][0];
    }
    pts[n - 1] = polys[0][0];

    if (vbrk[id].empty()) {
      // get the points after discrete flow map
      discreteFlowMap(v, pts, tn, dt);

      // split long edges
      Vector<unsigned int> splitids =
          splitLongEdges(v, pts, knots, polys, tn, dt);

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
           << " Max chdlength: " << *maxp << " Min chdlength: " << *minp
           << endl;
    } else {
      int nks = vbrk[id].size();
      Vector<Point> respts;
      SimplicialComplex reskinks;
      int pre, pos;
      unsigned int supn = n;
      int splitnum = 0, removenum = 0;

      auto localtimestep = [&](const int pre,
                               const int pos,
                               unsigned int &back) {
        Vector<Point> localpts(std::next(pts.begin(), pre),
                               std::next(pts.begin(), pos + 1));
        const Vector<Polynomial<Order, Point>> localpolys(
            std::next(polys.begin(), pre), std::next(polys.begin(), pos));
        const Vector<Real> localknots(std::next(knots.begin(), pre),
                                      std::next(knots.begin(), pos + 1));
        unsigned int localn = localpts.size();

        // get the points after discrete flow map
        discreteFlowMap(v, localpts, tn, dt);

        // split long edges
        Vector<unsigned int> splitids =
            splitLongEdges(v, localpts, localknots, localpolys, tn, dt);
        splitnum += splitids.size();

        // remove small edges
        Vector<unsigned int> removeids = removeSmallEdges(localpts);
        removenum += removeids.size();

        if (!respts.empty())
          respts.pop_back();

        respts.insert(respts.end(), localpts.begin(), localpts.end());

        supn += localpts.size() - localn;

        back += localpts.size() - 1;
        if (back != supn - 1) {
          reskinks.insert(Simplex{std::initializer_list<unsigned int>{back}});
          newkinks.push_back(std::make_pair((unsigned int)id, back));
        }
        return;
      };

      unsigned int back = 0;
      pre = 0;
      if (vbrk[id][0] != 0) {
        pos = vbrk[id][0];
        localtimestep(pre, pos, back);
      } else {
        reskinks.insert(Simplex{std::initializer_list<unsigned int>{back}});
        newkinks.push_back(std::make_pair((unsigned int)id, back));
      }

      pre = vbrk[id][0];
      int i = 1;
      while (i < nks) {
        pos = vbrk[id][i];
        localtimestep(pre, pos, back);
        pre = pos;
        i++;
      }
      pos = n - 1;
      localtimestep(pre, pos, back);

      int num = respts.size();
      Vector<Real> dist(num - 1);

      for (int i = 0; i < num - 1; i++) {
        dist[i] = norm(respts[i + 1] - respts[i], 2);
      }

      auto maxp = max_element(dist.begin(), dist.end());
      auto minp = min_element(dist.begin(), dist.end());

      cout << "Curve " << id << ":  Add " << splitnum << " Points,"
           << " remove " << removenum << " Points."
           << " Max chdlength: " << *maxp << " Min chdlength: " << *minp
           << endl;

      crv.define(respts, reskinks);
    }
  }
  ys = YS(SegmentedRealizableSpadjor<Order>(vcrv), tol);
  ys.setKinks(newkinks);
  return;
}

// VectorOnHypersurface version, with unique difference in splitLongEdges.

template <int Order>
void MARS2DIMV<Order, VectorOnHypersurface>::discreteFlowMap(
    const VectorOnHypersurface<2> &v,
    Vector<Point> &pts,
    Real tn,
    Real dt) {
  Base::TI->timeStep(v, pts, tn, dt);
  return;
}

template <int Order>
Vector<unsigned int> MARS2DIMV<Order, VectorOnHypersurface>::removeSmallEdges(
    Vector<Point> &pts) {
  int num = pts.size();
  Vector<unsigned int> ids(num);
  for (int i = 0; i < num; i++) {
    ids[i] = i;
  }

  while (true) {
    // output ids, debug use
    /*
    std::cout << "(";
    for (auto &id : ids)
    {
        std::cout << id << ", ";
    }
    std::cout << "\b)" << std::endl;
    */

    if (!removeIMV(ids, pts, (chdLenRange.lo())[0]))
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
bool splitIMV(Vector<bool> &ids,
              Vector<Point> &oldpts,
              const Vector<Real> &knots,
              Vector<Polynomial<Order, Point>> &polys,
              const Vector<Real> &dist,
              Real highBound) {
  int num = oldpts.size();
  Vector<Point> res(2 * num);
  Vector<bool> resid(2 * num);
  Polynomial<Order, Point> lpoly;

  auto addpt = [&](int &count, Point pt, bool type) {
    res[count] = pt;
    resid[count] = type;
    count++;
    if (count >= (int)res.size()) {
      res.resize(2 * count);
      resid.resize(2 * count);
    }
    return;
  };

  int count = 0;
  addpt(count, oldpts[0], ids[0]);

  for (int i = 0; i < num - 1; i++) {
    if (dist[i] <= highBound) {
      addpt(count, oldpts[i + 1], ids[i + 1]);
      continue;
    }

    int N = ceil(dist[i] / highBound);
    Real dt = (knots[i + 1] - knots[i]) / N;
    lpoly = polys[i];

    for (int j = 1; j < N; j++) {
      addpt(count, lpoly(dt * j), true);
    }
    addpt(count, oldpts[i + 1], ids[i + 1]);
  }
  res.resize(count);
  resid.resize(count);
  oldpts = res;
  ids = resid;
  return count != num;
}

template <int Order>
Vector<unsigned int> MARS2DIMV<Order, VectorOnHypersurface>::splitLongEdges(
    const VectorOnHypersurface<2> &v,
    Vector<Point> &pts,
    const Vector<Real> &knots,
    Vector<Polynomial<Order, Point>> &polys,
    Real tn,
    Real dt) {
  int num = pts.size();
  assert(num == (int)knots.size() && num == (int)polys.size() + 1);

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
    if (!splitIMV(ids, oldpts, knots, polys, dist, (chdLenRange.hi())[0]))
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
void MARS2DIMV<Order, VectorOnHypersurface>::timeStep(
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
    Vector<Real> knots = crv.getKnots();
    Vector<Point> pts(polys.size() + 1);
    for (int i = 0; i < (int)polys.size(); i++) {
      pts[i] = polys[i][0];
    }
    pts[polys.size()] = polys[0][0];

    // get the points after discrete flow map
    discreteFlowMap(v, pts, tn, dt);

    // split long edges
    Vector<unsigned int> splitids =
        splitLongEdges(v, pts, knots, polys, tn, dt);

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

template class MARS2DIMV<2, VectorFunction>;
template class MARS2DIMV<4, VectorFunction>;

template class MARS2DIMV<2, VectorOnHypersurface>;
template class MARS2DIMV<4, VectorOnHypersurface>;
