#include "Polygon.h"

std::vector<Polygon::rVec> Polygon::generateInnerPoint(int n) const {
  Real tol = YSB::Tolerance::Instance()->getTol();
  PointsLocater plocator(tol);
  CPolygon polygon;
  for (auto &&e : vecE) {
    polygon.push_back(CPoint(e.p[0][0], e.p[0][1]));
  }
  CDT cdt;
  cdt.insert_constraint(
      polygon.vertices_begin(), polygon.vertices_end(), true);
  std::vector<YSB::Triangle<Real, 2>> triangulation;
  for (CFacehandle f : cdt.finite_face_handles()) {
    YSB::Point<Real, 2> p0{f->vertex(0)->point()[0], f->vertex(0)->point()[1]};
    YSB::Point<Real, 2> p1{f->vertex(1)->point()[0], f->vertex(1)->point()[1]};
    YSB::Point<Real, 2> p2{f->vertex(2)->point()[0], f->vertex(2)->point()[1]};

    rVec midp1{(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0};
    rVec midp2{(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0};
    rVec midp3{(p2[0] + p0[0]) / 2.0, (p2[1] + p0[1]) / 2.0};
    std::vector<rVec> vecMidP{midp1, midp2, midp3};
    auto pos = plocator(vecE, vecMidP, bounded);
    int count = 0;
    for (int apos : pos) {
      if (apos == -2 * bounded + 1)
        count++;
    }

    if (count == 0)
      triangulation.push_back(YSB::Triangle<Real, 2>{p0, p1, p2});
  }

  std::vector<rVec> pts(n);

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(0.2, 0.6);
  std::uniform_int_distribution<> disti(0, triangulation.size() - 1);
  for (int i = 0; i < n; i++) {
    int id = disti(eng);
    auto tri = triangulation[id];
    rVec p(0.0);
    Real co[3];
    for (int j = 0; j < 3; j++) {
      co[j] = distr(eng);
      p[0] += tri.vertex(j)[0] * co[j];
      p[1] += tri.vertex(j)[1] * co[j];
    }
    p = p / (co[0] + co[1] + co[2]);
    pts[i] = p;
  }
  return pts;
}

void Polygon::correctOrientation(
    std::vector<YSB::Triangle<Real, 2>> &vecTri) const {
  std::vector<YSB::Triangle<Real, 2>> newVecTri;
  std::set<YSB::Segment<Real, 2>, YSB::SegmentCompare> bdry;
  std::map<YSB::Segment<Real, 2>, std::vector<int>, YSB::SegmentCompare>
      eNeighbor;
  YSB::PointCompare pCmp;
  for (auto &&e : vecE)
    bdry.insert(YSB::Segment<Real, 2>(YSB::Point<Real, 2>(0.0) + e.p[0],
                                      YSB::Point<Real, 2>(0.0) + e.p[1]));
  std::vector<bool> record(vecTri.size(), true);

  // assure the triangulation has no self-intersection points.
  std::map<YSB::Segment<Real, 2>, std::vector<int>, YSB::SegmentCompare>
      tempENeighbor;
  for (size_t i = 0; i < vecTri.size(); i++) {
    auto &tri = vecTri[i];
    for (int e = 0; e < 3; e++)
      tempENeighbor[tri.edge(e)].push_back(i);
  }
  for (auto &&en : tempENeighbor) {
    if (bdry.find(en.first) != bdry.end()) {
      assert(en.second.size() == 1 &&
             "The triangulation has self-intersection point!");
    } else {
      assert(en.second.size() == 2 &&
             "The triangulation has self-intersection point!");
    }
  }

  while (newVecTri.size() < vecTri.size()) {
    for (size_t i = 0; i < vecTri.size(); i++) {
      if (record[i] == false)
        continue;
      auto &tri = vecTri[i];
      for (int j = 0; j < 3; j++) {
        auto pos = bdry.find(tri.edge(j));
        if (pos != bdry.end()) {
          if (pCmp.compare(tri.edge(j)[0], (*pos)[0]) == 0) {
            newVecTri.push_back(tri);
            eNeighbor[tri.edge(j)].push_back(i);
            eNeighbor[tri.edge((j + 1) % 3)].push_back(i);
            eNeighbor[tri.edge((j + 2) % 3)].push_back(i);
          } else {
            tri.reverseOrientation();
            newVecTri.push_back(tri);
            eNeighbor[tri.edge(j)].push_back(i);
            eNeighbor[tri.edge((j + 1) % 3)].push_back(i);
            eNeighbor[tri.edge((j + 2) % 3)].push_back(i);
          }
          record[i] = false;
          break;
        } else {
          auto pos1 = eNeighbor.find(tri.edge(j));
          if (pos1 != eNeighbor.end()) {
            assert(pos1->second.size() == 1);
            if (pCmp.compare(tri.edge(j)[0], (pos1->first)[0]) == 0) {
              tri.reverseOrientation();
              newVecTri.push_back(tri);
              eNeighbor[tri.edge(j)].push_back(i);
              eNeighbor[tri.edge((j + 1) % 3)].push_back(i);
              eNeighbor[tri.edge((j + 2) % 3)].push_back(i);
            } else {
              newVecTri.push_back(tri);
              eNeighbor[tri.edge(j)].push_back(i);
              eNeighbor[tri.edge((j + 1) % 3)].push_back(i);
              eNeighbor[tri.edge((j + 2) % 3)].push_back(i);
            }
            record[i] = false;
            break;
          }
        }
      }
    }
  }
  vecTri = newVecTri;
}

std::vector<YSB::Triangle<Real, 2>> Polygon::triangulate(
    const std::vector<rVec> &pts) const {
  Real tol = YSB::Tolerance::Instance()->getTol();
  PointsLocater plocator(tol);
  CPolygon polygon;
  for (auto &&e : vecE) {
    polygon.push_back(CPoint(e.p[0][0], e.p[0][1]));
  }
  CDT cdt;
  cdt.insert_constraint(
      polygon.vertices_begin(), polygon.vertices_end(), true);
  std::vector<CPoint> cpts;
  for (auto &&p : pts) {
    cpts.push_back(CPoint(p[0], p[1]));
  }
  cdt.insert(cpts.begin(), cpts.end());
  std::vector<YSB::Triangle<Real, 2>> res;
  for (CFacehandle f : cdt.finite_face_handles()) {
    YSB::Point<Real, 2> p0{f->vertex(0)->point()[0], f->vertex(0)->point()[1]};
    YSB::Point<Real, 2> p1{f->vertex(1)->point()[0], f->vertex(1)->point()[1]};
    YSB::Point<Real, 2> p2{f->vertex(2)->point()[0], f->vertex(2)->point()[1]};

    rVec midp1{(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0};
    rVec midp2{(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0};
    rVec midp3{(p2[0] + p0[0]) / 2.0, (p2[1] + p0[1]) / 2.0};
    std::vector<rVec> vecMidP{midp1, midp2, midp3};
    auto pos = plocator(vecE, vecMidP, bounded);
    int count = 0;
    for (int apos : pos) {
      if (apos == -2 * bounded + 1)
        count++;
    }

    if (count == 0)
      res.push_back(YSB::Triangle<Real, 2>{p0, p1, p2});
  }

  correctOrientation(res);

  YSB::SegmentCompare segCmp;
  YSB::PointCompare pCmp;
  std::map<YSB::Segment<Real, 2>, std::vector<int>, YSB::SegmentCompare>
      neighbor;
  for (size_t i = 0; i < res.size(); i++) {
    auto tri = res[i];
    for (int j = 0; j < 3; j++) {
      neighbor[tri.edge(j)].push_back(i);
    }
  }
  for (auto an : neighbor) {
    if (an.second.size() != 2) {
      assert(an.second.size() == 1);
      for (auto bd : vecE) {
        auto nbd = YSB::Segment<Real, 2>(YSB::Point<Real, 2>(0.0) + bd.p[0],
                                         YSB::Point<Real, 2>(0.0) + bd.p[1]);
        if (segCmp.compare(nbd, an.first) == 0) {
          if (pCmp.compare(nbd[0], an.first[0]) != 0) {
            return std::vector<YSB::Triangle<Real, 2>>();
          }
        }
      }
      continue;
    }
    auto tri1 = res[an.second[0]];
    YSB::Segment<Real, 2> edge1;
    for (int i = 0; i < 3; i++)
      if (segCmp.compare(tri1.edge(i), an.first) == 0)
        edge1 = tri1.edge(i);
    auto tri2 = res[an.second[1]];
    YSB::Segment<Real, 2> edge2;
    for (int i = 0; i < 3; i++)
      if (segCmp.compare(tri2.edge(i), an.first) == 0)
        edge2 = tri2.edge(i);
    if (pCmp.compare(edge1[0], edge2[0]) == 0) {
      return std::vector<YSB::Triangle<Real, 2>>();
    }
  }

  return res;
}

Real Polygon::Area() const {
  Real area = 0;
  for (auto &&e : vecE) {
    area += 0.5 * (e.p[0][0] + e.p[1][0]) * (e.p[1][1] - e.p[0][1]);
  }
  return std::abs(area);
}