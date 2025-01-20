#include "LinearMARS3D.h"

#include "EdgeSwapper.h"
#include "VertexAdjustor.h"

#include <algorithm>
#include <queue>

template <int Order>
void LinearMARS3D<Order, VectorFunction>::discreteFlowMap(
    const VectorFunction<3> &v,
    YSB::YinSet<3, 2> &ys,
    Real tn,
    Real k) {
  for (auto &gsurf : ys.boundaryRef().vecGluedSurfaceRef()) {
    auto vecTri = gsurf.vecTriangle();
    for (auto &tri : vecTri) {
      YSB::Point<Real, 3> vecp[3] = {
          tri.vertex(0), tri.vertex(1), tri.vertex(2)};
      for (int i = 0; i < 3; i++) {
        Vec<Real, 3> p{vecp[i][0], vecp[i][1], vecp[i][2]};
        p = Base::TI->timeStep(v, p, tn, k);
        vecp[i] = YSB::Point<Real, 3>{p[0], p[1], p[2]};
      }
      tri = YSB::Triangle<Real, 3>{vecp[0], vecp[1], vecp[2]};
    }
    gsurf = YSB::GluedSurface<Real, 2>(vecTri);
  }
}

template <>
void LinearMARS3D<2, VectorFunction>::splitLongEdges(
    const VectorFunction<3> &v,
    YSB::Triangle<Real, 3> &tri,
    YSB::Triangle<Real, 3> &tritn,
    std::map<YSB::Triangle<Real, 3>, bool, YSB::TriangleCompare> &record,
    Real tn,
    Real k) {
  if (tri.edge(tri.maxEdgeID()).length() > chdLenRange.hi()[0])
    record[tri] = false;
  else
    return;

  if (tri.edge(tri.minEdgeID()).length() > chdLenRange.hi()[0]) {
    YSB::Point<Real, 3> midptn[3];
    YSB::Point<Real, 3> midp[3];
    for (int i = 0; i < 3; i++) {
      midptn[i] = YSB::Point<Real, 3>{
          (tritn.vertex(i)[0] + tritn.vertex((i + 1) % 3)[0]) / 2.0,
          (tritn.vertex(i)[1] + tritn.vertex((i + 1) % 3)[1]) / 2.0,
          (tritn.vertex(i)[2] + tritn.vertex((i + 1) % 3)[2]) / 2.0};
      Vec<Real, 3> midvtn{midptn[i][0], midptn[i][1], midptn[i][2]};
      auto midv = Base::TI->timeStep(v, midvtn, tn, k);
      midp[i] = YSB::Point<Real, 3>{midv[0], midv[1], midv[2]};
    }

    YSB::Triangle<Real, 3> subtritn1{tritn.vertex(0), midptn[0], midptn[2]};
    YSB::Triangle<Real, 3> subtri1{tri.vertex(0), midp[0], midp[2]};
    record[subtri1] = true;
    splitLongEdges(v, subtri1, subtritn1, record, tn, k);

    YSB::Triangle<Real, 3> subtritn2{midptn[0], tritn.vertex(1), midptn[1]};
    YSB::Triangle<Real, 3> subtri2{midp[0], tri.vertex(1), midp[1]};
    record[subtri2] = true;
    splitLongEdges(v, subtri2, subtritn2, record, tn, k);

    YSB::Triangle<Real, 3> subtritn3{midptn[2], midptn[1], tritn.vertex(2)};
    YSB::Triangle<Real, 3> subtri3{midp[2], midp[1], tri.vertex(2)};
    record[subtri3] = true;
    splitLongEdges(v, subtri3, subtritn3, record, tn, k);

    YSB::Triangle<Real, 3> subtritn4{midptn[0], midptn[1], midptn[2]};
    YSB::Triangle<Real, 3> subtri4{midp[0], midp[1], midp[2]};
    record[subtri4] = true;
    splitLongEdges(v, subtri4, subtritn4, record, tn, k);
  } else {
    int i = tri.maxEdgeID();
    YSB::Point<Real, 3> midptn{
        (tritn.vertex(i)[0] + tritn.vertex((i + 1) % 3)[0]) / 2.0,
        (tritn.vertex(i)[1] + tritn.vertex((i + 1) % 3)[1]) / 2.0,
        (tritn.vertex(i)[2] + tritn.vertex((i + 1) % 3)[2]) / 2.0};
    Vec<Real, 3> midvtn{midptn[0], midptn[1], midptn[2]};
    auto midv = Base::TI->timeStep(v, midvtn, tn, k);
    YSB::Point<Real, 3> midp{midv[0], midv[1], midv[2]};

    YSB::Triangle<Real, 3> subtritn1{
        tritn.vertex(i), midptn, tritn.vertex((i + 2) % 3)};
    YSB::Triangle<Real, 3> subtri1{
        tri.vertex(i), midp, tri.vertex((i + 2) % 3)};
    record[subtri1] = true;
    splitLongEdges(v, subtri1, subtritn1, record, tn, k);

    YSB::Triangle<Real, 3> subtritn2{
        midptn, tritn.vertex((i + 1) % 3), tritn.vertex((i + 2) % 3)};
    YSB::Triangle<Real, 3> subtri2{
        midp, tri.vertex((i + 1) % 3), tri.vertex((i + 2) % 3)};
    record[subtri2] = true;
    splitLongEdges(v, subtri2, subtritn2, record, tn, k);
  }
}

template <>
void LinearMARS3D<2, VectorFunction>::splitLongEdges(
    const VectorFunction<3> &v,
    YSB::YinSet<3, 2> &ys,
    const YSB::YinSet<3, 2> &ystn,
    Real tn,
    Real k) {
  for (size_t i = 0; i < ys.boundary().vecGluedSurface().size(); i++) {
    auto &gsurf = ys.boundary().vecGluedSurface()[i];
    auto &gsurftn = ystn.boundary().vecGluedSurface()[i];
    std::vector<YSB::Triangle<Real, 3>> res;
    for (size_t j = 0; j < gsurf.vecTriangle().size(); j++) {
      std::map<YSB::Triangle<Real, 3>, bool, YSB::TriangleCompare> record;
      YSB::Triangle<Real, 3> tri = gsurf.vecTriangle()[j];
      YSB::Triangle<Real, 3> tritn = gsurftn.vecTriangle()[j];
      record[tri] = true;
      splitLongEdges(v, tri, tritn, record, tn, k);
      for (auto it = record.begin(); it != record.end(); ++it) {
        if (it->second == true)
          res.push_back(it->first);
      }
    }
    ys.boundaryRef().vecGluedSurfaceRef()[i] = YSB::GluedSurface<Real, 2>(res);
  }
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::splitLongEdgesHighOrder(
    const VectorFunction<3> &v,
    YSB::Triangle<Real, 3> &tri,
    YSB::Triangle<Real, 3> &tritn,
    YSB::Triangle<Real, 3> &triOriginaltn,
    std::map<YSB::Triangle<Real, 3>, bool, YSB::TriangleCompare> &record,
    const PolynomialSurface<2> &triPolySurf,
    std::map<YSB::Segment<Real, 3>, PolynomialSurface<2>, YSB::SegmentCompare>
        &edgePolySurf,
    Real tn,
    Real k) {
  if (tri.edge(tri.maxEdgeID()).length() > chdLenRange.hi()[0])
    record[tri] = false;
  else
    return;

  YSB::PointCompare pCmp;
  if (tri.edge(tri.minEdgeID()).length() > chdLenRange.hi()[0]) {
    YSB::Point<Real, 3> midptn[3];
    YSB::Point<Real, 3> midproptn[3];
    YSB::Point<Real, 3> midp[3];
    for (int i = 0; i < 3; i++) {
      midptn[i] = YSB::Point<Real, 3>{
          (tritn.vertex(i)[0] + tritn.vertex((i + 1) % 3)[0]) / 2.0,
          (tritn.vertex(i)[1] + tritn.vertex((i + 1) % 3)[1]) / 2.0,
          (tritn.vertex(i)[2] + tritn.vertex((i + 1) % 3)[2]) / 2.0};
      bool onLineRecord = false;
      for (int e = 0; e < 3; e++) {
        if (triOriginaltn.edge(e).locatePoint(midptn[i]) !=
            YSB::Segment<Real, 3>::Outer) {
          midproptn[i] =
              edgePolySurf[triOriginaltn.edge(e)].projectToSurface(midptn[i]);
          onLineRecord = true;
          break;
        }
      }
      if (onLineRecord == false) {
        midproptn[i] = triPolySurf.projectToSurface(midptn[i]);
      }

      Vec<Real, 3> midprovtn{
          midproptn[i][0], midproptn[i][1], midproptn[i][2]};
      auto midv = Base::TI->timeStep(v, midprovtn, tn, k);
      midp[i] = YSB::Point<Real, 3>{midv[0], midv[1], midv[2]};
    }

    YSB::Triangle<Real, 3> subtritn1{tritn.vertex(0), midptn[0], midptn[2]};
    YSB::Triangle<Real, 3> subtri1{tri.vertex(0), midp[0], midp[2]};
    record[subtri1] = true;
    splitLongEdgesHighOrder(v,
                            subtri1,
                            subtritn1,
                            triOriginaltn,
                            record,
                            triPolySurf,
                            edgePolySurf,
                            tn,
                            k);

    YSB::Triangle<Real, 3> subtritn2{midptn[0], tritn.vertex(1), midptn[1]};
    YSB::Triangle<Real, 3> subtri2{midp[0], tri.vertex(1), midp[1]};
    record[subtri2] = true;
    splitLongEdgesHighOrder(v,
                            subtri2,
                            subtritn2,
                            triOriginaltn,
                            record,
                            triPolySurf,
                            edgePolySurf,
                            tn,
                            k);

    YSB::Triangle<Real, 3> subtritn3{midptn[2], midptn[1], tritn.vertex(2)};
    YSB::Triangle<Real, 3> subtri3{midp[2], midp[1], tri.vertex(2)};
    record[subtri3] = true;
    splitLongEdgesHighOrder(v,
                            subtri3,
                            subtritn3,
                            triOriginaltn,
                            record,
                            triPolySurf,
                            edgePolySurf,
                            tn,
                            k);

    YSB::Triangle<Real, 3> subtritn4{midptn[0], midptn[1], midptn[2]};
    YSB::Triangle<Real, 3> subtri4{midp[0], midp[1], midp[2]};
    record[subtri4] = true;
    splitLongEdgesHighOrder(v,
                            subtri4,
                            subtritn4,
                            triOriginaltn,
                            record,
                            triPolySurf,
                            edgePolySurf,
                            tn,
                            k);
  } else {
    int i = tri.maxEdgeID();
    YSB::Point<Real, 3> midptn{
        (tritn.vertex(i)[0] + tritn.vertex((i + 1) % 3)[0]) / 2.0,
        (tritn.vertex(i)[1] + tritn.vertex((i + 1) % 3)[1]) / 2.0,
        (tritn.vertex(i)[2] + tritn.vertex((i + 1) % 3)[2]) / 2.0};
    YSB::Point<Real, 3> midproptn;
    bool onLineRecord = false;
    for (int e = 0; e < 3; e++) {
      if (triOriginaltn.edge(e).locatePoint(midptn) !=
          YSB::Segment<Real, 3>::Outer) {
        midproptn =
            edgePolySurf[triOriginaltn.edge(e)].projectToSurface(midptn);
        onLineRecord = true;
        break;
      }
    }
    if (onLineRecord == false) {
      midproptn = triPolySurf.projectToSurface(midptn);
    }

    Vec<Real, 3> midprovtn{midproptn[0], midproptn[1], midproptn[2]};
    auto midv = Base::TI->timeStep(v, midprovtn, tn, k);
    YSB::Point<Real, 3> midp{midv[0], midv[1], midv[2]};

    YSB::Triangle<Real, 3> subtritn1{
        tritn.vertex(i), midptn, tritn.vertex((i + 2) % 3)};
    YSB::Triangle<Real, 3> subtri1{
        tri.vertex(i), midp, tri.vertex((i + 2) % 3)};
    record[subtri1] = true;
    splitLongEdgesHighOrder(v,
                            subtri1,
                            subtritn1,
                            triOriginaltn,
                            record,
                            triPolySurf,
                            edgePolySurf,
                            tn,
                            k);

    YSB::Triangle<Real, 3> subtritn2{
        midptn, tritn.vertex((i + 1) % 3), tritn.vertex((i + 2) % 3)};
    YSB::Triangle<Real, 3> subtri2{
        midp, tri.vertex((i + 1) % 3), tri.vertex((i + 2) % 3)};
    record[subtri2] = true;
    splitLongEdgesHighOrder(v,
                            subtri2,
                            subtritn2,
                            triOriginaltn,
                            record,
                            triPolySurf,
                            edgePolySurf,
                            tn,
                            k);
  }
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::splitLongEdges(
    const VectorFunction<3> &v,
    YSB::YinSet<3, 2> &ys,
    const YSB::YinSet<3, 2> &ystn,
    Real tn,
    Real k) {
  for (size_t i = 0; i < ys.boundary().vecGluedSurface().size(); i++) {
    auto &gsurf = ys.boundary().vecGluedSurface()[i];
    auto &gsurftn = ystn.boundary().vecGluedSurface()[i];
    auto &vecTri = gsurf.vecTriangle();
    auto &vecTritn = gsurftn.vecTriangle();

    std::map<YSB::Point<Real, 3>,
             std::set<YSB::Point<Real, 3>, YSB::PointCompare>,
             YSB::PointCompare>
        pNeighbor;
    for (size_t j = 0; j < vecTritn.size(); j++) {
      auto tritn = vecTritn[j];
      for (int k = 0; k < 3; k++) {
        pNeighbor[tritn.vertex(k)].insert(tritn.vertex((k + 1) % 3));
        pNeighbor[tritn.vertex(k)].insert(tritn.vertex((k + 2) % 3));
      }
    }

    std::map<YSB::Segment<Real, 3>, PolynomialSurface<2>, YSB::SegmentCompare>
        edgePolySurf;
    int coefNum = factorial(Order + 1) / (factorial(Order - 1) * 2) + Order;
    for (size_t j = 0; j < vecTritn.size(); j++) {
      auto tritn = vecTritn[j];
      auto tri = vecTri[j];
      for (int k = 0; k < 3; k++) {
        if (tri.edge(k).length() < chdLenRange.hi()[0])
          continue;
        if (edgePolySurf.find(tritn.edge(k)) != edgePolySurf.end())
          continue;
        std::set<YSB::Point<Real, 3>, YSB::PointCompare> setFitPts;
        std::queue<YSB::Point<Real, 3>> queuePts;
        queuePts.push(tritn.edge(k)[0]);
        queuePts.push(tritn.edge(k)[1]);
        while (setFitPts.size() < coefNum) {
          assert(queuePts.size() > 0 && "Do not have enough fitting points!");
          auto tempP = queuePts.front();
          queuePts.pop();
          auto insertInfo = setFitPts.insert(tempP);
          if (insertInfo.second == true) {
            for (auto &&np : pNeighbor[tempP]) {
              queuePts.push(np);
            }
          }
        }

        std::vector<YSB::Point<Real, 3>> vecFitPts;
        std::copy(setFitPts.begin(),
                  setFitPts.end(),
                  std::inserter(vecFitPts, vecFitPts.end()));
        edgePolySurf.insert({tritn.edge(k), PolynomialSurface<2>(vecFitPts)});
      }
    }

    std::vector<YSB::Triangle<Real, 3>> res;
    for (size_t j = 0; j < vecTri.size(); j++) {
      std::map<YSB::Triangle<Real, 3>, bool, YSB::TriangleCompare> record;
      YSB::Triangle<Real, 3> tri = vecTri[j];
      YSB::Triangle<Real, 3> tritn = vecTritn[j];
      if (tri.edge(tri.maxEdgeID()).length() <= chdLenRange.hi()[0]) {
        res.push_back(tri);
        continue;
      }

      std::set<YSB::Point<Real, 3>, YSB::PointCompare> setFitPts;
      std::queue<YSB::Point<Real, 3>> queuePts;
      queuePts.push(tritn.vertex(0));
      queuePts.push(tritn.vertex(1));
      queuePts.push(tritn.vertex(2));
      while (setFitPts.size() < coefNum) {
        assert(queuePts.size() > 0 && "Do not have enough fitting points!");
        auto tempP = queuePts.front();
        queuePts.pop();
        auto insertInfo = setFitPts.insert(tempP);
        if (insertInfo.second == true) {
          for (auto &&np : pNeighbor[tempP]) {
            queuePts.push(np);
          }
        }
      }
      assert(setFitPts.size() >= coefNum &&
             "Do not have enough fitting points!");

      std::vector<YSB::Point<Real, 3>> vecFitPts;
      std::copy(setFitPts.begin(),
                setFitPts.end(),
                std::inserter(vecFitPts, vecFitPts.end()));
      PolynomialSurface<2> triPolySurf(vecFitPts);

      record[tri] = true;
      splitLongEdgesHighOrder(
          v, tri, tritn, tritn, record, triPolySurf, edgePolySurf, tn, k);

      for (auto it = record.begin(); it != record.end(); ++it) {
        if (it->second == true)
          res.push_back(it->first);
      }
    }
    ys.boundaryRef().vecGluedSurfaceRef()[i] = YSB::GluedSurface<Real, 2>(res);
  }
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::removeSmallEdges(
    YSB::YinSet<3, 2> &ys) {
  auto &vecGSurf = ys.boundaryRef().vecGluedSurfaceRef();
  YSB::PointCompare pCmp;
  YSB::TriangleCompare triCmp;
  for (size_t i = 0; i < vecGSurf.size(); i++) {
    auto vecTri = vecGSurf[i].vecTriangle();
    while (1) {
      std::vector<bool> record(vecTri.size(), true);
      std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>
          segNeighbor;
      std::map<YSB::Point<Real, 3>, std::vector<int>, YSB::PointCompare>
          pNeighbor;
      for (size_t j = 0; j < vecTri.size(); j++) {
        auto tri = vecTri[j];
        for (int k = 0; k < 3; k++) {
          if (tri.edge(k).length() < chdLenRange.lo()[0]) {
            segNeighbor[tri.edge(k)].push_back(j);
          }
          pNeighbor[tri.vertex(k)].push_back(j);
        }
      }
      if (segNeighbor.size() < 1)
        break;

      std::set<YSB::Point<Real, 3>, YSB::PointCompare> adjustedPts;
      for (auto &&sn : segNeighbor) {
        bool ifmodified = false;
        for (auto triid : sn.second) {
          if (vecTri[triid].edgeID(sn.first) == -1) {
            ifmodified = true;
            break;
          }
        }
        if (ifmodified == true)
          continue;
        auto p1 = sn.first[0];
        auto p2 = sn.first[1];
        if (adjustedPts.find(p1) != adjustedPts.end() ||
            adjustedPts.find(p2) != adjustedPts.end())
          continue;

        adjustedPts.insert(p1);
        adjustedPts.insert(p2);
        for (auto triid : sn.second) {
          record[triid] = false;  // delete triangles next to the small edge
        }
        YSB::Point<Real, 3> midp{p1[0], p1[1], p1[2]};

        for (auto triid : pNeighbor[p1]) {
          if (record[triid] == false)
            continue;

          auto &tri = vecTri[triid];
          int k;
          for (k = 0; k < 3; k++) {
            if (pCmp.compare(p1, tri.vertex(k)) == 0)
              break;
          }
          tri = YSB::Triangle<Real, 3>{
              midp, tri.vertex((k + 1) % 3), tri.vertex((k + 2) % 3)};
        }
        for (auto triid : pNeighbor[p2]) {
          if (record[triid] == false)
            continue;
          auto &tri = vecTri[triid];
          int k;
          for (k = 0; k < 3; k++) {
            if (pCmp.compare(p2, tri.vertex(k)) == 0)
              break;
          }
          tri = YSB::Triangle<Real, 3>{
              midp, tri.vertex((k + 1) % 3), tri.vertex((k + 2) % 3)};
        }
      }

      std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare> setTri;
      for (size_t j = 0; j < vecTri.size(); j++) {
        if (record[j] == true) {
          auto insertInfo = setTri.insert(vecTri[j]);
          if (insertInfo.second == false)
            setTri.erase(insertInfo.first);
        }
      }
      vecTri.clear();
      std::copy(
          setTri.begin(), setTri.end(), std::inserter(vecTri, vecTri.end()));
    }
    ys.boundaryRef().vecGluedSurfaceRef()[i] =
        YSB::GluedSurface<Real, 2>(vecTri);
  }
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::swapEdges(YSB::YinSet<3, 2> &ys) {
  EdgeSwapper ES(chdLenRange.hi()[0], minAngle);
  ES.swapLocally(ys);
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::adjustVertex(YSB::YinSet<3, 2> &ys) {
  auto &vecGSurf = ys.boundaryRef().vecGluedSurfaceRef();
  std::vector<std::vector<int>> vecThinTriID(vecGSurf.size());
  for (size_t i = 0; i < vecGSurf.size(); i++) {
    for (size_t j = 0; j < vecGSurf[i].vecTriangle().size(); j++) {
      const auto &tri = vecGSurf[i].vecTriangle()[j];
      if (tri.minAngle() < minAngle)
        vecThinTriID[i].push_back(int(j));
    }
  }

  bool flag = false;  // true: need adjustment
  for (size_t i = 0; i < vecThinTriID.size(); i++) {
    if (vecThinTriID[i].size() != 0) {
      flag = true;
      break;
    }
  }
  if (flag == false)
    return;

  std::vector<std::vector<std::vector<int>>> vecPTriIDNeighbor(
      vecGSurf.size());
  std::vector<std::vector<std::vector<int>>> vecTriPIDNeighbor(
      vecGSurf.size());
  std::vector<std::vector<std::set<int>>> vecPPIDNeighbor(vecGSurf.size());
  YSB::PointCompare pCmp;

  for (size_t i = 0; i < vecGSurf.size(); i++) {
    if (vecThinTriID[i].size() == 0)
      continue;
    auto &gsurf = vecGSurf[i];
    std::map<YSB::Point<Real, 3>, std::vector<int>, YSB::PointCompare>
        pTriNeighbor;
    std::vector<YSB::Point<Real, 3>> vecP;

    for (size_t j = 0; j < gsurf.vecTriangle().size(); j++) {
      const auto &tri = gsurf.vecTriangle()[j];
      for (int k = 0; k < 3; k++) {
        if (pTriNeighbor.find(tri.vertex(k)) == pTriNeighbor.end()) {
          pTriNeighbor[tri.vertex(k)] = std::vector<int>{int(j)};
          vecP.push_back(tri.vertex(k));
        } else {
          pTriNeighbor[tri.vertex(k)].push_back(j);
        }
      }
    }

    std::vector<std::vector<int>> pTriIDNeighbor(vecP.size());
    std::vector<std::vector<int>> triPIDNeighbor(gsurf.vecTriangle().size(),
                                                 std::vector<int>(3));
    std::vector<std::set<int>> pPIDNeighbor(vecP.size());

    for (size_t j = 0; j < vecP.size(); j++) {
      pTriIDNeighbor[j] = pTriNeighbor[vecP[j]];
      for (int triID : pTriIDNeighbor[j]) {
        const auto &tri = gsurf.vecTriangle()[triID];
        for (int k = 0; k < 3; k++) {
          if (pCmp.compare(tri.vertex(k), vecP[j]) == 0)
            triPIDNeighbor[triID][k] = j;
        }
      }
    }
    for (auto tripid : triPIDNeighbor) {
      for (int k = 0; k < 3; k++) {
        pPIDNeighbor[tripid[k]].insert(tripid[(k + 1) % 3]);
        pPIDNeighbor[tripid[k]].insert(tripid[(k + 2) % 3]);
      }
    }
    vecPTriIDNeighbor[i] = pTriIDNeighbor;
    vecTriPIDNeighbor[i] = triPIDNeighbor;
    vecPPIDNeighbor[i] = pPIDNeighbor;
  }

  std::vector<std::vector<YSB::Triangle<Real, 3>>> vecVecTri(vecGSurf.size());
  for (size_t i = 0; i < vecVecTri.size(); i++) {
    vecVecTri[i] = vecGSurf[i].vecTriangle();
  }

  LinearEnergyFunctor<Real> EF;
  VertexAdjustor<Order> VA(&EF, chdLenRange, minAngle);
  for (size_t i = 0; i < vecThinTriID.size(); i++) {
    auto triPIDNeighbor = vecTriPIDNeighbor[i];
    auto pTriIDNeighbor = vecPTriIDNeighbor[i];
    auto pPIDNeighbor = vecPPIDNeighbor[i];
    for (size_t j = 0; j < vecThinTriID[i].size(); j++) {
      auto id = vecThinTriID[i][j];
      const auto &thinTri = vecVecTri[i][id];
      if (thinTri.minAngle() > minAngle)
        continue;
      auto maxedgeID = thinTri.maxEdgeID();
      auto maxV = triPIDNeighbor[id][(maxedgeID + 2) %
                                     3];  // the vertex of the maximum angle
      std::queue<int> queueP;
      std::set<int> setP;
      queueP.push(maxV);
      setP.insert(maxV);
      std::set<int> setTriID;
      bool res = false;
      int minNumPts =
          factorial(Order + 1) / (factorial(Order - 1) * 2) + Order;
      for (int iter = 0; iter < 5; iter++) {
        do {
          int n = queueP.size();
          for (int j = 0; j < n; j++) {
            assert(!queueP.empty() && "do not have enough points!");
            auto p = queueP.front();
            queueP.pop();
            for (auto np : pPIDNeighbor[p]) {
              auto info = setP.insert(np);
              if (info.second == true)
                queueP.push(np);
            }
            for (auto triID : pTriIDNeighbor[p]) {
              setTriID.insert(triID);
            }
          }
        } while (setP.size() < minNumPts);

        std::vector<YSB::Triangle<Real, 3>> vecTri;
        for (auto triID : setTriID) {
          vecTri.push_back(vecVecTri[i][triID]);
        }
        std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> bdry;
        YSB::SurfacePatch<Real, 2> sp(vecTri, bdry);
        // res = VA.adjustLocally(sp,
        // sqrt(chdLenRange.lo()[0]*chdLenRange.hi()[0]));
        res = VA.adjustLocally(sp, chdLenRange.lo()[0]);
        if (res == true) {
          int count = 0;
          for (auto triID : setTriID) {
            vecVecTri[i][triID] = sp.vecTriangle()[count++];
          }
          break;
        }
      }
    }
    vecGSurf[i] = YSB::GluedSurface<Real, 2>(vecVecTri[i]);
  }
}

bool isOrientedTri(std::vector<YSB::Triangle<Real, 3>> &vecTri) {
  bool res = true;
  YSB::PointCompare pCmp;

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
      res = false;
      break;
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

bool isOrientedSp(const YSB::SurfacePatch<Real, 2> &sp) {
  bool res = true;
  YSB::PointCompare pCmp;

  std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>
      neighbor;
  for (size_t i = 0; i < sp.vecTriangle().size(); i++) {
    auto tri = sp.vecTriangle()[i];
    for (int j = 0; j < 3; j++) {
      neighbor[tri.edge(j)].push_back(i);
    }
  }
  for (auto an : neighbor) {
    if (an.second.size() == 1)
      continue;
    if (an.second.size() > 2) {
      res = false;
      break;
    }
    auto tri1 = sp.vecTriangle()[an.second[0]];
    auto edge1 = tri1.edge(tri1.edgeID(an.first));
    auto tri2 = sp.vecTriangle()[an.second[1]];
    auto edge2 = tri2.edge(tri2.edgeID(an.first));
    if (pCmp.compare(edge1[0], edge2[0]) == 0) {
      res = false;
      break;
    }
  }

  return res;
}

Real minVecTriAngle(const std::vector<YSB::Triangle<Real, 3>> &vecTri) {
  Real res = 999.0;
  for (auto &&tri : vecTri) {
    if (tri.minAngle() < res)
      res = tri.minAngle();
  }
  return res;
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::regenerate(YSB::YinSet<3, 2> &ys,
                                                     int wrongid) {
  auto &vecGSurf = ys.boundaryRef().vecGluedSurfaceRef();
  LinearEnergyFunctor<Real> EF;
  VertexAdjustor<Order> VA(&EF, chdLenRange, minAngle);
  YSB::PointCompare pCmp;
  int maxSearchIter = 5, maxRegenerateIter = 5;
  for (auto &gsurf : vecGSurf) {
    auto vecTri = gsurf.vecTriangle();

    std::map<YSB::Point<Real, 3>,
             std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare>,
             YSB::PointCompare>
        pTriNeighbor;
    std::map<YSB::Point<Real, 3>,
             std::set<YSB::Point<Real, 3>, YSB::PointCompare>,
             YSB::PointCompare>
        pPNeighbor;

    for (size_t i = 0; i < vecTri.size(); i++) {
      auto tri = vecTri[i];
      for (int j = 0; j < 3; j++) {
        pTriNeighbor[tri.vertex(j)].insert(tri);
        pPNeighbor[tri.vertex(j)].insert(tri.vertex((j + 1) % 3));
        pPNeighbor[tri.vertex(j)].insert(tri.vertex((j + 2) % 3));
      }
    }

    std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare> setTri;
    std::copy(
        vecTri.begin(), vecTri.end(), std::inserter(setTri, setTri.end()));
    std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare> hardTriSet;
    while (1) {
      bool record = true;

      YSB::Triangle<Real, 3> thinTri{YSB::Point<Real, 3>{0.0, 0.0, 0.0},
                                     YSB::Point<Real, 3>{0.0, 1.0, 0.0},
                                     YSB::Point<Real, 3>{0.0, 0.0, 1.0}};
      for (auto &&tri : setTri) {
        if (tri.minAngle() < minAngle ||
            tri.edge(tri.maxEdgeID()).length() > chdLenRange.hi()[0] ||
            tri.edge(tri.minEdgeID()).length() < chdLenRange.lo()[0]) {
          if (hardTriSet.find(tri) != hardTriSet.end())
            continue;
          record = false;
          thinTri = tri;
          break;
        }
      }
      if (record == true)
        break;

      auto maxV = thinTri.vertex((thinTri.maxEdgeID() + 2) % 3);
      std::queue<YSB::Point<Real, 3>> queueP;
      std::set<YSB::Point<Real, 3>, YSB::PointCompare> setP;
      std::queue<YSB::Point<Real, 3>> queueTempP;
      std::set<YSB::Point<Real, 3>, YSB::PointCompare> setTempP;
      queueP.push(maxV);
      setP.insert(maxV);
      queueTempP.push(maxV);
      setTempP.insert(maxV);
      bool res = false;
      int minNumPts =
          factorial(Order + 1) / (factorial(Order - 1) * 2) + Order;
      // get enough points for surface fitting
      do {
        int n = queueTempP.size();
        for (int j = 0; j < n; j++) {
          assert(!queueTempP.empty() && "do not have enough points!");
          auto p = queueTempP.front();
          queueTempP.pop();
          for (auto &&np : pPNeighbor[p]) {
            auto insertInfo = setTempP.insert(np);
            if (insertInfo.second == true)
              queueTempP.push(np);
          }
        }
      } while (setTempP.size() < std::max(minNumPts, 9));

      std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare> setAdjustingTri;
      for (int iter = 0; iter < maxSearchIter; iter++) {
        assert(!queueP.empty() && "do not have enough points!");
        auto p = queueP.front();
        queueP.pop();
        for (auto &&np : pPNeighbor[p]) {
          auto insertInfo = setP.insert(np);
          if (insertInfo.second == true)
            queueP.push(np);
        }
        for (auto &&ntri : pTriNeighbor[p]) {
          setAdjustingTri.insert(ntri);
        }

        std::vector<YSB::Triangle<Real, 3>> localVecTri;
        std::copy(setAdjustingTri.begin(),
                  setAdjustingTri.end(),
                  std::inserter(localVecTri, localVecTri.end()));

        std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> bdry;
        Real maxminAngle = -1.0;
        std::vector<YSB::Triangle<Real, 3>> maxminAngleVecTri;

        auto adjustNeighborRelation =
            [](const std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare>
                   &SetAdjustingTri,
               const std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare>
                   &SetAdjustedTri,
               std::map<YSB::Point<Real, 3>,
                        std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare>,
                        YSB::PointCompare> &PTriNeighbor,
               std::map<YSB::Point<Real, 3>,
                        std::set<YSB::Point<Real, 3>, YSB::PointCompare>,
                        YSB::PointCompare> &PPNeighbor) {
              std::set<YSB::Point<Real, 3>, YSB::PointCompare> setInnerPts;
              std::set<YSB::Point<Real, 3>, YSB::PointCompare> setBdryPts;
              std::map<YSB::Segment<Real, 3>,
                       std::vector<YSB::Triangle<Real, 3>>,
                       YSB::SegmentCompare>
                  edgeNeighbor;
              for (auto &&tri : SetAdjustingTri) {
                for (int e = 0; e < 3; e++) {
                  edgeNeighbor[tri.edge(e)].push_back(tri);
                  setInnerPts.insert(tri.vertex(e));
                }
              }
              for (auto &&en : edgeNeighbor) {
                if (en.second.size() == 1) {
                  setBdryPts.insert(en.first[0]);
                  setBdryPts.insert(en.first[1]);
                }
              }
              for (auto &&p : setBdryPts)
                setInnerPts.erase(p);

              for (auto &&p : setInnerPts) {
                PTriNeighbor.erase(p);
                PPNeighbor.erase(p);
              }
              for (auto &&p : setBdryPts) {
                auto tempSetTri = PTriNeighbor[p];
                for (auto &&tri : PTriNeighbor[p]) {
                  if (SetAdjustingTri.find(tri) != SetAdjustingTri.end())
                    tempSetTri.erase(tri);
                }
                PTriNeighbor[p] = tempSetTri;

                auto tempSetP = PPNeighbor[p];
                for (auto &&q : PPNeighbor[p]) {
                  if (setInnerPts.find(q) != setInnerPts.end())
                    tempSetP.erase(q);
                }
                PPNeighbor[p] = tempSetP;
              }

              for (auto &&tri : SetAdjustedTri) {
                for (int e = 0; e < 3; e++) {
                  PTriNeighbor[tri.vertex(e)].insert(tri);
                  PPNeighbor[tri.vertex(e)].insert(tri.vertex((e + 1) % 3));
                  PPNeighbor[tri.vertex(e)].insert(tri.vertex((e + 2) % 3));
                }
              }
            };

        for (int k = 0; k < maxRegenerateIter; k++) {
          YSB::SurfacePatch<Real, 2> sp(localVecTri, bdry);
          VA.regenerateSurfacePatch(sp, setTempP, wrongid);
          if (sp.vecTriangle().size() == 0) {
            if (k == maxRegenerateIter - 1 && iter == maxSearchIter - 1) {
              hardTriSet.insert(thinTri);
            }
            // else
            continue;
          }
          Real minSPAngle = minVecTriAngle(sp.vecTriangle());
          if (minSPAngle > maxminAngle) {
            maxminAngleVecTri = sp.vecTriangle();
          }

          res = isRegular(sp);
          if (res == true) {
            std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare>
                setAdjustedTri;
            std::copy(sp.vecTriangle().begin(),
                      sp.vecTriangle().end(),
                      std::inserter(setAdjustedTri, setAdjustedTri.end()));

            adjustNeighborRelation(
                setAdjustingTri, setAdjustedTri, pTriNeighbor, pPNeighbor);
            for (auto &&tri : setAdjustingTri)
              setTri.erase(tri);
            std::copy(setAdjustedTri.begin(),
                      setAdjustedTri.end(),
                      std::inserter(setTri, setTri.end()));

            auto ori = isOrientedSp(sp);
            if (ori == false) {
              YSB::SurfacePatch<Real, 2> wrongsp(sp.vecTriangle(), bdry);
            }
            assert(ori == true);

            break;
          }
          if (res == false && iter == maxSearchIter - 1 &&
              k == maxRegenerateIter - 1) {
            std::set<YSB::Triangle<Real, 3>, YSB::TriangleCompare>
                setAdjustedTri;
            std::copy(maxminAngleVecTri.begin(),
                      maxminAngleVecTri.end(),
                      std::inserter(setAdjustedTri, setAdjustedTri.end()));

            adjustNeighborRelation(
                setAdjustingTri, setAdjustedTri, pTriNeighbor, pPNeighbor);

            for (auto &&tri : setAdjustingTri)
              setTri.erase(tri);
            std::copy(setAdjustedTri.begin(),
                      setAdjustedTri.end(),
                      std::inserter(setTri, setTri.end()));

            YSB::SurfacePatch<Real, 2> tempSp(maxminAngleVecTri, bdry);
            auto ori = isOrientedSp(tempSp);
            assert(ori == true);
            break;
          }
        }
        if (res == true)
          break;
      }
    }
    vecTri.clear();
    std::copy(
        setTri.begin(), setTri.end(), std::inserter(vecTri, vecTri.end()));
    gsurf = YSB::GluedSurface<Real, 2>(vecTri);
  }
}

template <int Order>
bool LinearMARS3D<Order, VectorFunction>::isRegular(
    YSB::SurfacePatch<Real, 2> &sp) {
  for (auto tri : sp.vecTriangle()) {
    for (int i = 0; i < 3; i++) {
      if (tri.edge(i).length() < chdLenRange.lo()[0] ||
          tri.edge(i).length() > chdLenRange.hi()[0])
        return false;
    }
    if (tri.minAngle() < minAngle)
      return false;
  }
  return true;
}

template <int Order>
bool LinearMARS3D<Order, VectorFunction>::isOriented(YSB::YinSet<3, 2> &ys) {
  bool res = true;
  YSB::PointCompare pCmp;
  for (const auto &gs : ys.boundary().vecGluedSurface()) {
    std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>
        neighbor;
    for (size_t i = 0; i < gs.vecTriangle().size(); i++) {
      auto tri = gs.vecTriangle()[i];
      for (int j = 0; j < 3; j++) {
        neighbor[tri.edge(j)].push_back(i);
      }
    }
    for (auto an : neighbor) {
      if (an.second.size() != 2) {
        res = false;
        break;
      }
      auto tri1 = gs.vecTriangle()[an.second[0]];
      auto edge1 = tri1.edge(tri1.edgeID(an.first));
      auto tri2 = gs.vecTriangle()[an.second[1]];
      auto edge2 = tri2.edge(tri2.edgeID(an.first));
      if (pCmp.compare(edge1[0], edge2[0]) == 0) {
        res = false;
        break;
      }
    }
  }
  return res;
}

template <int Order>
int LinearMARS3D<Order, VectorFunction>::countSmallAngle(
    const std::vector<YSB::Triangle<Real, 3>> &vecTri) {
  int res = 0;
  for (auto &&tri : vecTri) {
    if (tri.minAngle() < minAngle)
      res++;
  }
  return res;
}

bool existSingleEdge(YSB::YinSet<3, 2> &ys) {
  const auto &vecGSurf = ys.boundary().vecGluedSurface();
  std::vector<
      std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>>
      segNeighbor(vecGSurf.size());
  for (size_t i = 0; i < vecGSurf.size(); i++) {
    const auto &vecTri = vecGSurf[i].vecTriangle();
    for (size_t j = 0; j < vecTri.size(); j++) {
      const auto &tri = vecTri[j];
      for (int k = 0; k < 3; k++) {
        auto iter = segNeighbor[i].find(tri.edge(k));
        if (iter != segNeighbor[i].end())
          iter->second.push_back(j);
        else {
          segNeighbor[i][tri.edge(k)] = std::vector<int>{int(j)};
        }
      }
    }
  }
  for (size_t i = 0; i < vecGSurf.size(); i++) {
    for (auto &&en : segNeighbor[i]) {
      if (en.second.size() != 2)
        assert(false && "There is a single edge!");
    }
  }
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::timeStep(const VectorFunction<3> &v,
                                                   YSB::YinSet<3, 2> &ys,
                                                   Real tn,
                                                   Real dt,
                                                   int *numProcessed) {
  auto GS = ys.boundary().vecGluedSurface();
  YSB::YinSetBoundary<3, 2> ystnbdry(GS);
  YSB::YinSet<3, 2> ystn(std::move(ystnbdry));

  discreteFlowMap(v, ys, tn, dt);

  splitLongEdges(v, ys, ystn, tn, dt);

  removeSmallEdges(ys);

  int numSmallAngle1 =
      countSmallAngle(ys.boundary().vecGluedSurface()[0].vecTriangle());
  swapEdges(ys);

  int numSmallAngle2 =
      countSmallAngle(ys.boundary().vecGluedSurface()[0].vecTriangle());
  adjustVertex(ys);

  int numSmallAngle3 =
      countSmallAngle(ys.boundary().vecGluedSurface()[0].vecTriangle());
  regenerate(ys, int(tn / dt + 1));

  numProcessed[0] = numSmallAngle1 - numSmallAngle2;
  numProcessed[1] = numSmallAngle2 - numSmallAngle3;
  numProcessed[2] = numSmallAngle3;
}

template <int Order>
void LinearMARS3D<Order, VectorFunction>::timeStep(const VectorFunction<3> &v,
                                                   YSB::YinSet<3, 2> &ys,
                                                   Real tn,
                                                   Real dt) {
  auto GS = ys.boundary().vecGluedSurface();
  YSB::YinSetBoundary<3, 2> ystnbdry(GS);
  YSB::YinSet<3, 2> ystn(std::move(ystnbdry));

  discreteFlowMap(v, ys, tn, dt);

  splitLongEdges(v, ys, ystn, tn, dt);

  removeSmallEdges(ys);

  swapEdges(ys);

  adjustVertex(ys);

  regenerate(ys, int(tn / dt + 1));
}

template class LinearMARS3D<3, VectorFunction>;
template class LinearMARS3D<2, VectorFunction>;
template class LinearMARS3D<4, VectorFunction>;