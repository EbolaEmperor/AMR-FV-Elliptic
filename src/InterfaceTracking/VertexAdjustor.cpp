#include "InterfaceTracking/VertexAdjustor.h"

#include "InterfaceTracking/EdgeSwapper.h"
#include "InterfaceTracking/Polygon.h"
#include "InterfaceTracking/PolynomialSurface.h"
#include "YinSet3D/PointCompare.h"
#include "YinSet3D/SegmentCompare.h"

#include <eigen3/Eigen/Eigen>
#include <queue>

template <int Order>
bool VertexAdjustor<Order>::adjustLocally(YSB::SurfacePatch<Real, 2> &sp,
                                          Real R) {
  std::vector<YSB::Triangle<Real, 3>> vecTri = sp.vecTriangle();
  std::vector<YSB::Point<Real, 3>> pts;
  std::vector<std::vector<int>> ptsID;
  std::set<YSB::Point<Real, 3>, YSB::PointCompare> tempPts;
  YSB::PointCompare pCmp;
  for (const auto &tri : vecTri) {
    std::vector<int> aID(3);
    for (int i = 0; i < 3; i++) {
      auto insertInfo = tempPts.insert(tri.vertex(i));
      if (insertInfo.second == true) {
        pts.push_back(tri.vertex(i));
        aID[i] = pts.size() - 1;
      } else {
        for (size_t j = 0; j < pts.size(); j++) {
          if (pCmp.compare(tri.vertex(i), pts[j]) == 0)
            aID[i] = j;
        }
      }
    }
    ptsID.push_back(aID);
  }

  std::vector<std::set<int>> pNeighbor(pts.size(), std::set<int>());
  for (size_t i = 0; i < vecTri.size(); i++) {
    auto aID = ptsID[i];
    for (int j = 0; j < 3; j++) {
      pNeighbor[aID[j]].insert(aID[(j + 1) % 3]);
      pNeighbor[aID[j]].insert(aID[(j + 2) % 3]);
    }
  }

  std::vector<bool> fixedRecord(pts.size(), false);
  std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>
      edgeNeighbor;
  for (size_t i = 0; i < vecTri.size(); i++) {
    for (int j = 0; j < 3; j++) {
      edgeNeighbor[vecTri[i].edge(j)].push_back(i);
    }
  }
  Real mediumBdryLength = 0;
  int bdryCount = 0;
  for (auto iter = edgeNeighbor.begin(); iter != edgeNeighbor.end(); ++iter) {
    if (iter->second.size() != 2) {
      mediumBdryLength += iter->first.length();
      bdryCount++;
      int edgeid = vecTri[iter->second[0]].edgeID(iter->first);
      fixedRecord[ptsID[iter->second[0]][edgeid % 3]] = true;
      fixedRecord[ptsID[iter->second[0]][(edgeid + 1) % 3]] = true;
    }
  }
  // R = mediumBdryLength / bdryCount;
  R = 0.1 * chdLenRange.hi()[0];
  // R = 0;

  // for each point, use BFS to find a certain number of points (just id)
  // around it to fit a surface.
  std::vector<std::set<int>> vecFittingPtsID(pts.size());
  int minNumPts = factorial(Order + 1) / (factorial(Order - 1) * 2) + Order;
  for (size_t i = 0; i < pts.size(); i++) {
    if (fixedRecord[i] == true)
      continue;
    std::set<int> aFittingPts;
    std::queue<int> ptsQueue;
    ptsQueue.push(i);
    while (aFittingPts.size() < minNumPts && ptsQueue.size() > 0) {
      int pt = ptsQueue.front();
      ptsQueue.pop();
      auto insertInfo = aFittingPts.insert(pt);
      if (insertInfo.second == true) {
        for (auto &&id : pNeighbor[pt]) {
          ptsQueue.push(id);
        }
      }
    }
    if (ptsQueue.size() < 1 && aFittingPts.size() < minNumPts)
      return false;
    // assert("The surface patch does not have enough points.");
    vecFittingPtsID[i] = aFittingPts;
  }

  std::vector<std::vector<std::pair<int, int>>> vecLink(pts.size());
  for (auto &&aID : ptsID) {
    for (int j = 0; j < 3; j++) {
      vecLink[aID[j]].push_back(
          std::make_pair(aID[(j + 1) % 3], aID[(j + 2) % 3]));
    }
  }
  std::vector<PolynomialSurface<Order - 1>> vecPolySurf(pts.size());
  for (size_t i = 0; i < pts.size(); i++) {
    if (fixedRecord[i] == true)
      continue;
    std::vector<YSB::Point<Real, 3>> fittingPts;
    for (auto id : vecFittingPtsID[i])
      fittingPts.push_back(pts[id]);
    vecPolySurf[i] = PolynomialSurface<Order - 1>(fittingPts);
  }
  int iter = 1;
  bool ifsucceed = false;
  while (iter <= maxLocalIter) {
    std::vector<Vec<Real, 3>> vecDE(pts.size());
    std::vector<Vec<Real, 3>> vecOffset(pts.size());
    for (size_t i = 0; i < pts.size(); i++) {
      if (fixedRecord[i] == true)
        continue;

      PolynomialSurface<Order - 1> localSurface = vecPolySurf[i];
      Vec<Real, 3> offset(0.0);
      YSB::Point<Real, 3> p1 = pts[i];
      for (int j : pNeighbor[i]) {
        YSB::Point<Real, 3> p2 = pts[j];
        offset = offset - (p1 - p2) / norm(p1 - p2) * 2.0 / pts.size() *
                              EF->calculateEnergyDerivative(norm(p1 - p2), R);
      }
      vecDE[i] = offset;
      YSB::Plane<Real> proPla(p1, localSurface.getNormalVector(p1));
      YSB::Point<Real, 3> tempP = p1 + offset;
      Vec<Real, 3> proOffset;
      if (proPla.contains(tempP)) {
        proOffset = offset;
      } else {
        YSB::Line<Real, 3> l(tempP, proPla.normalVec());
        YSB::Point<Real, 3> proTempP = proPla.intersect(l);
        proOffset = proTempP - p1;
      }
      vecOffset[i] = proOffset;
    }

    Real lenOffset = 0;
    for (size_t i = 0; i < pts.size(); i++) {
      lenOffset += pow(norm(vecOffset[i]), 2);
    }
    lenOffset = sqrt(lenOffset);

    Real dETOffset = 0;  // compute grad E dot offset
    Real stepsize = 1.0;
    for (size_t i = 0; i < pts.size(); i++) {
      vecOffset[i] = vecOffset[i] / lenOffset;
      dETOffset += -dot(vecDE[i], vecOffset[i]);

      Real minDist = 1e10;
      for (auto &&e : vecLink[i]) {
        Real aDist =
            norm(cross(pts[i] - pts[e.first], pts[e.first] - pts[e.second])) /
            norm(pts[e.first] - pts[e.second]);
        if (aDist < minDist)
          minDist = aDist;
      }
      if (stepsize * norm(vecOffset[i]) > minDist * alpha) {
        stepsize = minDist * alpha / norm(vecOffset[i]);
      }
    }

    Real oldE = 0;
    for (size_t i = 0; i < pts.size(); i++) {
      for (int j : pNeighbor[i]) {
        oldE += EF->calculateEnergy(norm(pts[i] - pts[j]), R);
      }
    }
    oldE /= pts.size();

    std::vector<YSB::Point<Real, 3>> tempPts = pts;
    for (int k = 0; k < 10; k++) {
      for (size_t i = 0; i < tempPts.size(); i++) {
        if (fixedRecord[i] == true)
          continue;
        tempPts[i] = pts[i] + vecOffset[i] * stepsize;
      }
      Real newE = 0;
      for (size_t i = 0; i < tempPts.size(); i++) {
        for (int j : pNeighbor[i]) {
          newE += EF->calculateEnergy(norm(tempPts[i] - tempPts[j]), R);
        }
      }
      newE /= tempPts.size();
      if (newE <= oldE + c * stepsize *
                             dETOffset)  // test if satisfying Armijo condition
      {
        break;
      }
      stepsize *= rho;
    }
    for (size_t i = 0; i < pts.size(); i++) {
      if (fixedRecord[i] == true)
        continue;
      pts[i] = vecPolySurf[i].projectToSurface(tempPts[i]);
    }
    // pts = tempPts;

    bool tag = true;
    for (auto &&aID : ptsID) {
      YSB::Triangle<Real, 3> temptri{pts[aID[0]], pts[aID[1]], pts[aID[2]]};
      for (int j = 0; j < 3; j++) {
        if (temptri.edge(j).length() > chdLenRange.hi()[0] ||
            temptri.edge(j).length() < chdLenRange.lo()[0]) {
          tag = false;
          break;
        }
      }
      if (temptri.minAngle() < minAngle)
        tag = false;
      if (tag == false)
        break;
    }
    if (tag == true) {
      ifsucceed = true;
      break;
    }
    iter++;
  }

  for (size_t i = 0; i < vecTri.size(); i++) {
    auto aID = ptsID[i];
    YSB::Triangle<Real, 3> tri{pts[aID[0]], pts[aID[1]], pts[aID[2]]};
    vecTri[i] = tri;
  }
  std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> setSegs;
  YSB::SurfacePatch<Real, 2> newsp(vecTri, setSegs);
  sp = std::move(newsp);

  return ifsucceed;
}

template <int Order>
void VertexAdjustor<Order>::projectOnTriangulation(
    const std::vector<YSB::Triangle<Real, 3>> &vecTri,
    YSB::Point<Real, 3> &ap) {
  Real tol = YSB::Tolerance::Instance()->getTol();
  std::set<YSB::Point<Real, 3>, YSB::PointCompare> setP;
  std::vector<YSB::Point<Real, 3>> vecP;
  for (const auto &tri : vecTri) {
    for (int i = 0; i < 3; i++)
      setP.insert(tri.vertex(i));
  }
  for (auto &&p : setP)
    vecP.push_back(p);

  YSB::Plane<Real> pla(vecP);

  std::vector<YSB::Triangle<Real, 3>> vecProTri;
  for (const auto &tri : vecTri) {
    YSB::Point<Real, 3> prov[3];
    for (int i = 0; i < 3; i++) {
      auto v = tri.vertex(i);
      if (pla.contains(v))
        prov[i] = v;
      else {
        YSB::Line<Real, 3> l(v, pla.normalVec());
        prov[i] = pla.intersect(l);
      }
    }
    vecProTri.push_back(YSB::Triangle<Real, 3>{prov[0], prov[1], prov[2]});
  }

  YSB::Point<Real, 3> pp;
  if (pla.contains(ap))
    pp = ap;
  else {
    YSB::Line<Real, 3> l(ap, pla.normalVec());
    pp = pla.intersect(l);
  }
  bool record = false;
  std::vector<std::array<Real, 3>> vecco;
  for (size_t j = 0; j < vecProTri.size(); j++) {
    std::array<Real, 3> co{-1, -1, -1};
    vecProTri[j].barycentric(pp, co);
    vecco.push_back(co);
    if (co[0] < -tol || co[1] < -tol || co[2] < -tol)
      continue;

    record = true;
    ap = vecTri[j].barycentric(co);
    break;
  }
}

template <>
bool VertexAdjustor<2>::adjustLocally(YSB::SurfacePatch<Real, 2> &sp, Real R) {
  std::vector<YSB::Triangle<Real, 3>> vecTri = sp.vecTriangle();
  std::vector<YSB::Point<Real, 3>> pts;
  std::vector<std::vector<int>> ptsID;
  std::set<YSB::Point<Real, 3>, YSB::PointCompare> tempPts;
  YSB::PointCompare pCmp;
  for (const auto &tri : vecTri) {
    std::vector<int> aID(3);
    for (int i = 0; i < 3; i++) {
      auto insertInfo = tempPts.insert(tri.vertex(i));
      if (insertInfo.second == true) {
        pts.push_back(tri.vertex(i));
        aID[i] = pts.size() - 1;
      } else {
        for (size_t j = 0; j < pts.size(); j++) {
          if (pCmp.compare(tri.vertex(i), pts[j]) == 0)
            aID[i] = j;
        }
      }
    }
    ptsID.push_back(aID);
  }

  auto originPts = pts;

  std::vector<std::set<int>> pNeighbor(pts.size(), std::set<int>());
  for (size_t i = 0; i < vecTri.size(); i++) {
    auto aID = ptsID[i];
    for (int j = 0; j < 3; j++) {
      pNeighbor[aID[j]].insert(aID[(j + 1) % 3]);
      pNeighbor[aID[j]].insert(aID[(j + 2) % 3]);
    }
  }

  std::vector<bool> fixedRecord(pts.size(), false);
  std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>
      edgeNeighbor;
  for (size_t i = 0; i < vecTri.size(); i++) {
    for (int j = 0; j < 3; j++) {
      edgeNeighbor[vecTri[i].edge(j)].push_back(i);
    }
  }
  Real mediumBdryLength = 0;
  int bdryCount = 0;
  for (auto iter = edgeNeighbor.begin(); iter != edgeNeighbor.end(); ++iter) {
    if (iter->second.size() != 2) {
      mediumBdryLength += iter->first.length();
      bdryCount++;
      int edgeid = vecTri[iter->second[0]].edgeID(iter->first);
      fixedRecord[ptsID[iter->second[0]][edgeid % 3]] = true;
      fixedRecord[ptsID[iter->second[0]][(edgeid + 1) % 3]] = true;
    }
  }
  R = 0.1 * chdLenRange.hi()[0];
  // R = 0;

  std::vector<std::vector<std::pair<int, int>>> vecLink(pts.size());
  for (auto &&aID : ptsID) {
    for (int j = 0; j < 3; j++) {
      vecLink[aID[j]].push_back(
          std::make_pair(aID[(j + 1) % 3], aID[(j + 2) % 3]));
    }
  }

  std::vector<std::vector<std::vector<int>>> pTriNeighbor(pts.size());
  for (size_t i = 0; i < ptsID.size(); i++) {
    auto aID = ptsID[i];
    for (int j = 0; j < 3; j++) {
      pTriNeighbor[aID[j]].push_back(aID);
    }
  }

  int iter = 1;
  bool ifsucceed = false;
  while (iter <= maxLocalIter) {
    std::vector<Vec<Real, 3>> vecDE(pts.size());
    std::vector<Vec<Real, 3>> vecOffset(pts.size());
    for (size_t i = 0; i < pts.size(); i++) {
      if (fixedRecord[i] == true)
        continue;
      Vec<Real, 3> offset(0.0);
      YSB::Point<Real, 3> p1 = pts[i];
      for (int j : pNeighbor[i]) {
        YSB::Point<Real, 3> p2 = pts[j];
        offset = offset - (p1 - p2) / norm(p1 - p2) * 2.0 / pts.size() *
                              EF->calculateEnergyDerivative(norm(p1 - p2), R);
      }
      vecDE[i] = offset;
      vecOffset[i] = offset;
    }

    Real lenOffset = 0;
    for (size_t i = 0; i < pts.size(); i++) {
      lenOffset += pow(norm(vecOffset[i]), 2);
    }
    lenOffset = sqrt(lenOffset);

    Real dETOffset = 0;  // compute grad E dot offset
    Real stepsize = 1.0;
    for (size_t i = 0; i < pts.size(); i++) {
      vecOffset[i] = vecOffset[i] / lenOffset;
      dETOffset += -dot(vecDE[i], vecOffset[i]);

      Real minDist = 1e10;
      for (auto &&e : vecLink[i]) {
        Real aDist =
            norm(cross(pts[i] - pts[e.first], pts[e.first] - pts[e.second])) /
            norm(pts[e.first] - pts[e.second]);
        if (aDist < minDist)
          minDist = aDist;
      }
      if (stepsize * norm(vecOffset[i]) > minDist * alpha) {
        stepsize = minDist * alpha / norm(vecOffset[i]);
      }
    }

    Real oldE = 0;
    for (size_t i = 0; i < pts.size(); i++) {
      for (int j : pNeighbor[i]) {
        oldE += EF->calculateEnergy(norm(pts[i] - pts[j]), R);
      }
    }
    oldE /= pts.size();

    std::vector<YSB::Point<Real, 3>> tempPts = pts;
    for (int k = 0; k < 10; k++) {
      for (size_t i = 0; i < tempPts.size(); i++) {
        if (fixedRecord[i] == true)
          continue;
        tempPts[i] = pts[i] + vecOffset[i] * stepsize;
      }
      Real newE = 0;
      for (size_t i = 0; i < tempPts.size(); i++) {
        for (int j : pNeighbor[i]) {
          newE += EF->calculateEnergy(norm(tempPts[i] - tempPts[j]), R);
        }
      }
      newE /= tempPts.size();
      if (newE <= oldE + c * stepsize *
                             dETOffset)  // test if satisfying Armijo condition
      {
        break;
      }
      stepsize *= rho;
    }

    for (size_t i = 0; i < tempPts.size(); i++) {
      if (fixedRecord[i])
        continue;
      std::vector<YSB::Triangle<Real, 3>> vecTempTri;
      for (auto &&aID : pTriNeighbor[i]) {
        vecTempTri.push_back(YSB::Triangle<Real, 3>{
            originPts[aID[0]], originPts[aID[1]], originPts[aID[2]]});
      }
      projectOnTriangulation(vecTempTri, tempPts[i]);
    }
    for (size_t i = 0; i < pts.size(); i++) {
      if (fixedRecord[i] == false)
        pts[i] = tempPts[i];
    }

    bool tag = true;
    for (auto &&aID : ptsID) {
      YSB::Triangle<Real, 3> temptri{pts[aID[0]], pts[aID[1]], pts[aID[2]]};
      for (int j = 0; j < 3; j++) {
        if (temptri.edge(j).length() > chdLenRange.hi()[0] ||
            temptri.edge(j).length() < chdLenRange.lo()[0]) {
          tag = false;
          break;
        }
      }
      if (temptri.minAngle() < minAngle)
        tag = false;
      if (tag == false)
        break;
    }
    if (tag == true) {
      ifsucceed = true;
      break;
    }
    iter++;
  }

  for (size_t i = 0; i < vecTri.size(); i++) {
    auto aID = ptsID[i];
    YSB::Triangle<Real, 3> tri{pts[aID[0]], pts[aID[1]], pts[aID[2]]};
    vecTri[i] = tri;
  }
  std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> setSegs;
  YSB::SurfacePatch<Real, 2> newsp(vecTri, setSegs);
  sp = std::move(newsp);
  return ifsucceed;
}

template <int Order>
void VertexAdjustor<Order>::adjustLocally2D(
    const Polygon &pg,
    std::vector<Vec<Real, 2>> &adjustedPts,
    Real R) {
  auto fixedPts = pg.getVertex();
  int iter = 1;
  size_t n = adjustedPts.size();
  size_t m = fixedPts.size();

  while (iter <= maxLocalIter) {
    std::vector<Vec<Real, 2>> vecOffset(n);
    for (size_t i = 0; i < n; i++) {
      Vec<Real, 2> offset(0.0);
      auto p1 = adjustedPts[i];
      for (size_t j = 0; j < n; j++) {
        if (j == i)
          continue;
        auto p2 = adjustedPts[j];
        offset = offset - (p1 - p2) / norm(p1 - p2) * 2.0 / (n * (n + m - 1)) *
                              EF->calculateEnergyDerivative(norm(p1 - p2), R);
      }
      for (size_t j = 0; j < m; j++) {
        auto p2 = fixedPts[j];
        offset = offset - (p1 - p2) / norm(p1 - p2) * 2.0 / (n * (n + m - 1)) *
                              EF->calculateEnergyDerivative(norm(p1 - p2), R);
      }

      vecOffset[i] = offset;
    }

    Real lenOffset = 0;
    for (size_t i = 0; i < n; i++) {
      lenOffset += pow(norm(vecOffset[i]), 2);
    }
    lenOffset = sqrt(lenOffset);
    if (lenOffset < 1e-8)
      break;

    Real dETOffset = 0;  // compute grad E dot offset
    Real stepsize = 1.0;
    for (size_t i = 0; i < n; i++) {
      vecOffset[i] = vecOffset[i] / lenOffset;
      dETOffset += -dot(vecOffset[i] * lenOffset, vecOffset[i]);

      Real minDist = 1e10;
      for (auto &&e : pg.getEdge()) {
        auto p1 = e.p[0], p2 = e.p[1];
        Real aDist = norm(cross(adjustedPts[i] - p1, p1 - p2)) / norm(p1 - p2);
        if (aDist < minDist)
          minDist = aDist;
      }
      if (stepsize * norm(vecOffset[i]) > minDist * alpha) {
        stepsize = minDist * alpha / norm(vecOffset[i]);
      }
    }

    Real oldE = 0;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        if (j == i)
          continue;
        oldE += EF->calculateEnergy(norm(adjustedPts[i] - adjustedPts[j]), R);
      }
      for (size_t j = 0; j < m; j++) {
        auto p2 = fixedPts[j];
        oldE += EF->calculateEnergy(norm(adjustedPts[i] - p2), R);
      }
    }
    oldE /= n * (n + m - 1);

    std::vector<Vec<Real, 2>> tempPts = adjustedPts;
    for (int k = 0; k < 10; k++) {
      for (size_t i = 0; i < tempPts.size(); i++) {
        tempPts[i] = adjustedPts[i] + vecOffset[i] * stepsize;
      }
      Real newE = 0;
      for (size_t i = 0; i < tempPts.size(); i++) {
        for (size_t j = 0; j < n; j++) {
          if (j == i)
            continue;
          oldE += EF->calculateEnergy(norm(tempPts[i] - tempPts[j]), R);
        }
        for (size_t j = 0; j < m; j++) {
          auto p2 = fixedPts[j];
          oldE += EF->calculateEnergy(norm(tempPts[i] - p2), R);
        }
      }
      newE /= n * (n + m - 1);
      if (newE <= oldE + c * stepsize *
                             dETOffset)  // test if satisfying Armijo condition
      {
        break;
      }
      stepsize *= rho;
    }
    adjustedPts = tempPts;
    iter++;
  }
}

bool isOriented(std::vector<YSB::Triangle<Real, 3>> &vecTri,
                std::vector<YSB::Segment<Real, 3>> &bdry) {
  bool res = true;
  YSB::PointCompare pCmp;
  YSB::SegmentCompare segCmp;

  std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>
      neighbor;
  for (size_t i = 0; i < vecTri.size(); i++) {
    auto tri = vecTri[i];
    for (int j = 0; j < 3; j++) {
      neighbor[tri.edge(j)].push_back(i);
    }
  }
  for (auto an : neighbor) {
    if (an.second.size() != 2) {
      assert(an.second.size() == 1);
      for (auto bd : bdry) {
        if (segCmp.compare(bd, an.first) == 0) {
          assert(pCmp.compare(bd[0], an.first[0]) == 0);
        }
      }
    }
    auto tri1 = vecTri[an.second[0]];
    auto edge1 = tri1.edge(tri1.edgeID(an.first));
    auto tri2 = vecTri[an.second[1]];
    auto edge2 = tri2.edge(tri2.edgeID(an.first));
    if (pCmp.compare(edge1[0], edge2[0]) == 0) {
      res = false;
      break;
    }
  }

  return res;
}

template <int Order>
Real VertexAdjustor<Order>::minVecTriAngle(
    const std::vector<YSB::Triangle<Real, 3>> &vecTri) {
  Real res = 999.0;
  for (auto &&tri : vecTri) {
    if (tri.minAngle() < res)
      res = tri.minAngle();
  }
  return res;
}

template <int Order>
Real VertexAdjustor<Order>::minVecTriAngle2D(
    const std::vector<YSB::Triangle<Real, 2>> &vecTri) {
  Real res = 999.0;
  for (auto &&tri : vecTri) {
    if (tri.minAngle() < res)
      res = tri.minAngle();
  }
  return res;
}

template <int Order>
void VertexAdjustor<Order>::regenerateSurfacePatch(
    YSB::SurfacePatch<Real, 2> &sp,
    std::set<YSB::Point<Real, 3>, YSB::PointCompare> setTempP,
    int wrongid) {
  EdgeSwapper ES(chdLenRange.hi()[0], minAngle);

  std::set<YSB::Point<Real, 3>, YSB::PointCompare> setP;
  std::vector<YSB::Point<Real, 3>> vecP;
  for (const auto &tri : sp.vecTriangle()) {
    for (int i = 0; i < 3; i++)
      setP.insert(tri.vertex(i));
  }
  for (auto &&p : setP)
    vecP.push_back(p);
  auto vecP1 = vecP;
  int minNumPts = factorial(Order + 1) / (factorial(Order - 1) * 2) + Order;
  if (vecP.size() < minNumPts) {
    vecP.clear();
    for (auto &&p : setTempP)
      vecP.push_back(p);
  }

  YSB::Plane<Real> pla(vecP1);
  auto normVec = pla.normalVec();
  auto fixPoint = pla.fixPoint();
  Real a = normVec[0], b = normVec[1], c = normVec[2];
  Real d1 = sqrt(pow(a, 2) + pow(b, 2));
  Real d2 = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
  Eigen::MatrixXd R(3, 3);
  R << b / d1, -a / d1, 0, a * c / (d1 * d2), b * c / (d1 * d2), -d1 / d2,
      a / d2, b / d2, c / d2;
  Eigen::Vector3d rn(a, b, c);
  rn = R * rn;
  Eigen::Vector3d rfp(fixPoint[0], fixPoint[1], fixPoint[2]);
  rfp = R * rfp;
  YSB::Plane<Real> rotatedPla(YSB::Point<Real, 3>{rfp(0), rfp(1), rfp(2)},
                              Vec<Real, 3>{rn(0), rn(1), rn(2)});
  std::vector<YSB::Point<Real, 3>> vecRP;
  for (auto &&p : vecP) {
    Eigen::Vector3d rp(p[0], p[1], p[2]);
    rp = R * rp;
    vecRP.emplace_back(YSB::Point<Real, 3>{rp(0), rp(1), rp(2)});
  }

  PolynomialSurface<Order - 1> rotatedPolySurf(vecRP, rotatedPla);

  std::map<YSB::Segment<Real, 3>,
           std::vector<YSB::Triangle<Real, 3>>,
           YSB::SegmentCompare>
      edgeNeighbor;
  for (const auto &tri : sp.vecTriangle()) {
    for (int i = 0; i < 3; i++)
      edgeNeighbor[tri.edge(i)].push_back(tri);
  }

  std::vector<Segment<2>> vecProRotatedBdry;
  std::map<YSB::Point<Real, 3>, YSB::Point<Real, 3>, YSB::PointCompare>
      fixBdry;

  for (auto &&en : edgeNeighbor) {
    if (en.second.size() == 1) {
      YSB::Segment<Real, 3> e = en.first;
      Eigen::Vector3d evp1(e[0][0], e[0][1], e[0][2]),
          evp2(e[1][0], e[1][1], e[1][2]);
      evp1 = R * evp1;
      evp2 = R * evp2;
      Segment<2> proRotatedE(Vec<Real, 2>{evp1(0), evp1(1)},
                             Vec<Real, 2>{evp2(0), evp2(1)});
      vecProRotatedBdry.push_back(proRotatedE);
      YSB::Point<Real, 3> p1{evp1(0), evp1(1), evp1(2)},
          p2{evp2(0), evp2(1), evp2(2)};
      p1 = rotatedPolySurf.projectToSurface(p1);
      p2 = rotatedPolySurf.projectToSurface(p2);
      fixBdry[p1] = e[0];
      fixBdry[p2] = e[1];
    }
  }

  Real tol = YSB::Tolerance::Instance()->getTol();
  VecCompare<Real, 2> vCmp(tol);
  bool isSimple = true;
  for (size_t i = 0; i < vecProRotatedBdry.size(); i++) {
    for (size_t j = i + 1; j < vecProRotatedBdry.size(); j++) {
      auto seg1 = vecProRotatedBdry[i];
      auto seg2 = vecProRotatedBdry[j];
      Vec<Real, 2> resv1, resv2;
      auto itsres = intersect(seg1, seg2, resv1, resv2, tol);
      if (itsres == Segment<2>::intsType::One) {
        if ((vCmp.compare(resv1, seg1.p[0]) != 0) &&
            (vCmp.compare(resv1, seg1.p[1]) != 0)) {
          isSimple = false;
        }
        if ((vCmp.compare(resv1, seg2.p[0]) != 0) &&
            (vCmp.compare(resv1, seg2.p[1]) != 0)) {
          isSimple = false;
        }
      }
    }
  }
  if (isSimple == false) {
    std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> bd;
    sp = YSB::SurfacePatch<Real, 2>(std::vector<YSB::Triangle<Real, 3>>(), bd);
    return;
  }

  Polygon pg(vecProRotatedBdry);

  // estimate the inserting point number
  int insertNum;
  Real averageEdgeLength = 0.0;
  for (auto &&e : vecProRotatedBdry) {
    averageEdgeLength += norm(e.p[1] - e.p[0]);
  }
  averageEdgeLength /= vecProRotatedBdry.size();
  Real pgArea = pg.Area();
  insertNum = std::max(
      0.0,
      std::round((4 * sqrt(3) * pgArea - 3 * pow(averageEdgeLength, 2) *
                                             (vecProRotatedBdry.size() - 2)) /
                 (6 * pow(averageEdgeLength, 2))));

  Real maxminAngle = -1.0;
  std::vector<YSB::Triangle<Real, 3>> maxminAngleVecTri;
  std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> bd;

  LogarithmEnergyFunctor<Real> EF;
  VertexAdjustor<Order> VAFor2D(&EF, chdLenRange, minAngle);

  for (int in = -1; in < 3; in++) {
    if (insertNum + in < 0)
      continue;
    Real maxminAngle2D = -1;
    std::vector<YSB::Triangle<Real, 2>> maxminAngleVecTri2D;
    for (int iter = 0; iter < 5; iter++) {
      auto insertPts = pg.generateInnerPoint(insertNum + in);
      VAFor2D.adjustLocally2D(pg, insertPts, averageEdgeLength);
      auto triangulation2D = pg.triangulate(insertPts);
      if (triangulation2D.size() == 0) {
        continue;
      }
      if (minVecTriAngle2D(triangulation2D) > maxminAngle2D) {
        maxminAngleVecTri2D = triangulation2D;
      }
    }
    if (maxminAngleVecTri2D.size() == 0)
      continue;
    auto proRotatedTriangulation = maxminAngleVecTri2D;

    std::vector<YSB::Triangle<Real, 3>> rotatedTriangulation;
    for (auto &&tri : proRotatedTriangulation) {
      std::vector<YSB::Point<Real, 3>> pts(3);
      for (int i = 0; i < 3; i++) {
        pts[i] = rotatedPolySurf.projectToSurface(
            YSB::Point<Real, 3>{tri.vertex(i)[0], tri.vertex(i)[1], 0.0});
      }
      rotatedTriangulation.push_back({pts[0], pts[1], pts[2]});
    }

    std::vector<YSB::Triangle<Real, 3>> newTriangulation;
    Eigen::MatrixXd RInverse = R.inverse();
    std::set<YSB::Point<Real, 3>, YSB::PointCompare> newsetP;
    for (auto &&tri : rotatedTriangulation) {
      std::vector<YSB::Point<Real, 3>> pts(3);
      for (int i = 0; i < 3; i++) {
        if (fixBdry.find(tri.vertex(i)) != fixBdry.end()) {
          pts[i] = fixBdry[tri.vertex(i)];
        } else {
          auto rp = tri.vertex(i);
          Eigen::Vector3d evp(rp[0], rp[1], rp[2]);
          evp = RInverse * evp;
          pts[i] = {evp(0), evp(1), evp(2)};
        }
      }
      for (int i = 0; i < 3; i++)
        newsetP.insert(pts[i]);
      newTriangulation.push_back({pts[0], pts[1], pts[2]});
    }
    sp = YSB::SurfacePatch<Real, 2>(newTriangulation, bd);
    for (int j = 0; j < 3; j++) {
      adjustLocally(sp, chdLenRange.lo()[0]);
      ES.swap(sp);
    }
    Real minSPAngle = minVecTriAngle(sp.vecTriangle());
    if (minSPAngle > maxminAngle) {
      maxminAngleVecTri = sp.vecTriangle();
    }
  }
  sp = YSB::SurfacePatch<Real, 2>(maxminAngleVecTri, bd);
}

template <>
void VertexAdjustor<2>::regenerateSurfacePatch(
    YSB::SurfacePatch<Real, 2> &sp,
    std::set<YSB::Point<Real, 3>, YSB::PointCompare> setTempP,
    int wrongid) {
  EdgeSwapper ES(chdLenRange.hi()[0], minAngle);

  std::set<YSB::Point<Real, 3>, YSB::PointCompare> setP;
  std::vector<YSB::Point<Real, 3>> vecP;
  for (const auto &tri : sp.vecTriangle()) {
    for (int i = 0; i < 3; i++)
      setP.insert(tri.vertex(i));
  }
  for (auto &&p : setP)
    vecP.push_back(p);
  auto vecP1 = vecP;
  if (vecP.size() < 9) {
    vecP.clear();
    for (auto &&p : setTempP)
      vecP.push_back(p);
  }

  YSB::Plane<Real> pla(vecP1);
  auto normVec = pla.normalVec();
  auto fixPoint = pla.fixPoint();
  Real a = normVec[0], b = normVec[1], c = normVec[2];
  Real d1 = sqrt(pow(a, 2) + pow(b, 2));
  Real d2 = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
  Eigen::MatrixXd R(3, 3);
  R << b / d1, -a / d1, 0, a * c / (d1 * d2), b * c / (d1 * d2), -d1 / d2,
      a / d2, b / d2, c / d2;
  Eigen::Vector3d rn(a, b, c);
  rn = R * rn;
  Eigen::Vector3d rfp(fixPoint[0], fixPoint[1], fixPoint[2]);
  rfp = R * rfp;
  YSB::Plane<Real> rotatedPla(YSB::Point<Real, 3>{rfp(0), rfp(1), rfp(2)},
                              Vec<Real, 3>{rn(0), rn(1), rn(2)});
  std::vector<YSB::Point<Real, 3>> vecRP;
  for (auto &&p : vecP) {
    Eigen::Vector3d rp(p[0], p[1], p[2]);
    rp = R * rp;
    vecRP.emplace_back(YSB::Point<Real, 3>{rp(0), rp(1), rp(2)});
  }

  PolynomialSurface<2> rotatedPolySurf(vecRP, rotatedPla);

  std::map<YSB::Segment<Real, 3>,
           std::vector<YSB::Triangle<Real, 3>>,
           YSB::SegmentCompare>
      edgeNeighbor;
  for (const auto &tri : sp.vecTriangle()) {
    for (int i = 0; i < 3; i++)
      edgeNeighbor[tri.edge(i)].push_back(tri);
  }

  std::vector<Segment<2>> vecProRotatedBdry;
  std::map<YSB::Point<Real, 3>, YSB::Point<Real, 3>, YSB::PointCompare>
      fixBdry;

  for (auto &&en : edgeNeighbor) {
    if (en.second.size() == 1) {
      YSB::Segment<Real, 3> e = en.first;
      Eigen::Vector3d evp1(e[0][0], e[0][1], e[0][2]),
          evp2(e[1][0], e[1][1], e[1][2]);
      evp1 = R * evp1;
      evp2 = R * evp2;
      Segment<2> proRotatedE(Vec<Real, 2>{evp1(0), evp1(1)},
                             Vec<Real, 2>{evp2(0), evp2(1)});
      vecProRotatedBdry.push_back(proRotatedE);
      YSB::Point<Real, 3> p1{evp1(0), evp1(1), evp1(2)},
          p2{evp2(0), evp2(1), evp2(2)};
      p1 = rotatedPolySurf.projectToSurface(p1);
      p2 = rotatedPolySurf.projectToSurface(p2);
      fixBdry[p1] = e[0];
      fixBdry[p2] = e[1];
    }
  }

  Real tol = YSB::Tolerance::Instance()->getTol();
  VecCompare<Real, 2> vCmp(tol);
  bool isSimple = true;
  for (size_t i = 0; i < vecProRotatedBdry.size(); i++) {
    for (size_t j = i + 1; j < vecProRotatedBdry.size(); j++) {
      auto seg1 = vecProRotatedBdry[i];
      auto seg2 = vecProRotatedBdry[j];
      Vec<Real, 2> resv1, resv2;
      auto itsres = intersect(seg1, seg2, resv1, resv2, tol);
      if (itsres == Segment<2>::intsType::One) {
        if ((vCmp.compare(resv1, seg1.p[0]) != 0) &&
            (vCmp.compare(resv1, seg1.p[1]) != 0)) {
          isSimple = false;
        }
        if ((vCmp.compare(resv1, seg2.p[0]) != 0) &&
            (vCmp.compare(resv1, seg2.p[1]) != 0)) {
          isSimple = false;
        }
      }
    }
  }
  if (isSimple == false) {
    std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> bd;
    sp = YSB::SurfacePatch<Real, 2>(std::vector<YSB::Triangle<Real, 3>>(), bd);
    return;
  }

  Polygon pg(vecProRotatedBdry);
  // estimate the inserting point number
  int insertNum;
  Real averageEdgeLength = 0.0;
  for (auto &&e : vecProRotatedBdry) {
    averageEdgeLength += norm(e.p[1] - e.p[0]);
  }
  averageEdgeLength /= vecProRotatedBdry.size();
  // averageEdgeLength = chdLenRange.lo()[0];
  Real pgArea = pg.Area();
  insertNum = std::max(
      0.0,
      std::round((4 * sqrt(3) * pgArea - 3 * pow(averageEdgeLength, 2) *
                                             (vecProRotatedBdry.size() - 2)) /
                 (6 * pow(averageEdgeLength, 2))));

  Real maxminAngle = -1.0;
  std::vector<YSB::Triangle<Real, 3>> maxminAngleVecTri;
  std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> bd;
  LogarithmEnergyFunctor<Real> EF;
  VertexAdjustor<2> VAFor2D(&EF, chdLenRange, minAngle);
  for (int in = -1; in < 3; in++) {
    if (insertNum + in < 0)
      continue;
    Real maxminAngle2D = -1;
    std::vector<YSB::Triangle<Real, 2>> maxminAngleVecTri2D;
    for (int iter = 0; iter < 5; iter++) {
      auto insertPts = pg.generateInnerPoint(insertNum + in);
      VAFor2D.adjustLocally2D(pg, insertPts, averageEdgeLength);
      auto triangulation2D = pg.triangulate(insertPts);
      if (triangulation2D.size() == 0) {
        continue;
      }
      if (minVecTriAngle2D(triangulation2D) > maxminAngle2D) {
        maxminAngleVecTri2D = triangulation2D;
      }
    }
    if (maxminAngleVecTri2D.size() == 0)
      continue;
    auto proRotatedTriangulation = maxminAngleVecTri2D;

    std::vector<YSB::Triangle<Real, 3>> rotatedTriangulation;
    for (auto &&tri : proRotatedTriangulation) {
      std::vector<YSB::Point<Real, 3>> pts(3);
      for (int i = 0; i < 3; i++) {
        pts[i] = rotatedPolySurf.projectToSurface(
            YSB::Point<Real, 3>{tri.vertex(i)[0], tri.vertex(i)[1], 0.0});
      }
      rotatedTriangulation.push_back({pts[0], pts[1], pts[2]});
    }

    std::vector<YSB::Triangle<Real, 3>> newTriangulation;
    Eigen::MatrixXd RInverse = R.inverse();
    for (auto &&tri : rotatedTriangulation) {
      std::vector<YSB::Point<Real, 3>> pts(3);
      for (int i = 0; i < 3; i++) {
        if (fixBdry.find(tri.vertex(i)) != fixBdry.end()) {
          pts[i] = fixBdry[tri.vertex(i)];
        } else {
          auto rp = tri.vertex(i);
          Eigen::Vector3d evp(rp[0], rp[1], rp[2]);
          evp = RInverse * evp;
          YSB::Point<Real, 3> tempp{evp(0), evp(1), evp(2)};
          std::vector<YSB::Point<Real, 3>> tempvp{tempp};
          // projectOnTriangulation(sp, tempvp);
          pts[i] = tempvp[0];
        }
      }
      newTriangulation.push_back({pts[0], pts[1], pts[2]});
    }
    auto tempsp = YSB::SurfacePatch<Real, 2>(newTriangulation, bd);
    for (int j = 0; j < 1; j++) {
      adjustLocally(tempsp, chdLenRange.lo()[0]);
      ES.swap(tempsp);
    }
    Real minSPAngle = minVecTriAngle(tempsp.vecTriangle());
    if (minSPAngle > maxminAngle) {
      maxminAngleVecTri = tempsp.vecTriangle();
    }
  }
  sp = YSB::SurfacePatch<Real, 2>(maxminAngleVecTri, bd);
}

template class VertexAdjustor<3>;
template class VertexAdjustor<2>;
template class VertexAdjustor<4>;