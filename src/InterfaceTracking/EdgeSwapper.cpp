#include "EdgeSwapper.h"

void EdgeSwapper::swap(YSB::SurfacePatch<Real, 2> &sp) {
  std::map<YSB::Segment<Real, 3>, std::vector<int>, YSB::SegmentCompare>
      segNeighbor;

  auto vecTri = sp.vecTriangle();
  for (size_t j = 0; j < vecTri.size(); j++) {
    auto tri = vecTri[j];
    for (int k = 0; k < 3; k++) {
      segNeighbor[tri.edge(k)].push_back(j);
    }
  }

  for (int iter = 0; iter < 5; iter++) {
    bool flag = false;

    int count = 0;
    for (size_t k = 0; k < vecTri.size(); k++) {
      const auto &tri = vecTri[k];
      if (tri.minAngle() < minAngle)
        count++;
    }

    for (size_t j = 0; j < vecTri.size(); j++) {
      auto thinTri1 = vecTri[j];
      int maxedgeID = thinTri1.maxEdgeID();
      auto maxedge = thinTri1.edge(maxedgeID);
      if (segNeighbor[maxedge].size() == 1)
        continue;
      assert(segNeighbor[maxedge].size() == 2 && "Non-manifold edge exists!");

      auto triID2 = segNeighbor[maxedge][0] == j ? segNeighbor[maxedge][1]
                                                 : segNeighbor[maxedge][0];
      auto thinTri2 = vecTri[triID2];
      int edgeID2 = thinTri2.edgeID(maxedge);
      Real oldminangle = std::min(thinTri1.minAngle(), thinTri2.minAngle());

      YSB::Triangle<Real, 3> newTri1{thinTri1.vertex((maxedgeID + 2) % 3),
                                     thinTri1.vertex(maxedgeID),
                                     thinTri2.vertex((edgeID2 + 2) % 3)};
      YSB::Triangle<Real, 3> newTri2{thinTri2.vertex((edgeID2 + 2) % 3),
                                     thinTri2.vertex(edgeID2),
                                     thinTri1.vertex((maxedgeID + 2) % 3)};

      Real newminangle = std::min(newTri1.minAngle(), newTri2.minAngle());
      YSB::Segment<Real, 3> newDiag(thinTri1.vertex((maxedgeID + 2) % 3),
                                    thinTri2.vertex((edgeID2 + 2) % 3));
      if (newDiag.length() > chdLenRange.hi()[0] ||
          newDiag.length() < chdLenRange.lo()[0] || oldminangle > newminangle)
        continue;

      // exclude the situation where the new triangle overlaps with the
      // original neighbor tiangle.
      auto neighborIDs1 = segNeighbor[newTri1.edge(0)];
      if (neighborIDs1.size() == 2) {
        int neighborTriID1 =
            (neighborIDs1[0] == j ? neighborIDs1[1] : neighborIDs1[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult1;
        vecTri[neighborTriID1].intersect(newTri1, tempresult1);
        if (tempresult1.size() > 2)
          continue;
      } else
        continue;
      auto neighborIDs2 = segNeighbor[newTri1.edge(1)];
      if (neighborIDs2.size() == 2) {
        int neighborTriID2 =
            (neighborIDs2[0] == triID2 ? neighborIDs2[1] : neighborIDs2[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult2;
        vecTri[neighborTriID2].intersect(newTri1, tempresult2);
        if (tempresult2.size() > 2)
          continue;
      } else
        continue;
      auto neighborIDs3 = segNeighbor[newTri2.edge(0)];
      if (neighborIDs3.size() == 2) {
        int neighborTriID3 =
            (neighborIDs3[0] == triID2 ? neighborIDs3[1] : neighborIDs3[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult3;
        vecTri[neighborTriID3].intersect(newTri2, tempresult3);
        if (tempresult3.size() > 2)
          continue;
      } else
        continue;
      auto neighborIDs4 = segNeighbor[newTri2.edge(1)];
      if (neighborIDs4.size() == 2) {
        int neighborTriID4 =
            (neighborIDs4[0] == j ? neighborIDs4[1] : neighborIDs4[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult4;
        vecTri[neighborTriID4].intersect(newTri2, tempresult4);
        if (tempresult4.size() > 2)
          continue;
      } else
        continue;

      flag = true;
      vecTri[j] = newTri1;
      vecTri[triID2] = newTri2;

      segNeighbor.erase(maxedge);
      segNeighbor[newDiag] = {int(j), triID2};
      if (segNeighbor[newTri1.edge(1)][0] == triID2)
        segNeighbor[newTri1.edge(1)][0] = int(j);
      else
        segNeighbor[newTri1.edge(1)][1] = int(j);
      if (segNeighbor[newTri2.edge(1)][0] == int(j))
        segNeighbor[newTri2.edge(1)][0] = triID2;
      else
        segNeighbor[newTri2.edge(1)][1] = triID2;
    }
    if (flag == false)
      break;
  }

  std::set<YSB::Segment<Real, 3>, YSB::SegmentCompare> bdry;
  sp = YSB::SurfacePatch<Real, 2>(vecTri, bdry);
}

void EdgeSwapper::swapLocally(YSB::YinSet<3, 2> &ys) {
  auto &vecGSurf = ys.boundaryRef().vecGluedSurfaceRef();
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

  std::vector<std::vector<YSB::Triangle<Real, 3>>> vecVecTri(vecGSurf.size());
  for (size_t i = 0; i < vecVecTri.size(); i++) {
    vecVecTri[i] = vecGSurf[i].vecTriangle();
  }

  for (size_t i = 0; i < vecGSurf.size(); i++) {
    int count = 0;
    for (size_t k = 0; k < vecGSurf[i].vecTriangle().size(); k++) {
      const auto &tri = vecGSurf[i].vecTriangle()[k];
      if (tri.minAngle() < minAngle)
        count++;
    }

    for (size_t j = 0; j < vecGSurf[i].vecTriangle().size(); j++) {
      auto thinTri1 = vecVecTri[i][j];
      if (thinTri1.minAngle() > minAngle)
        continue;

      int maxedgeID = thinTri1.maxEdgeID();
      auto maxedge = thinTri1.edge(maxedgeID);
      assert(segNeighbor[i][maxedge].size() == 2 &&
             "Non-manifold edge exists!");

      auto triID2 = segNeighbor[i][maxedge][0] == j
                        ? segNeighbor[i][maxedge][1]
                        : segNeighbor[i][maxedge][0];
      auto thinTri2 = vecVecTri[i][triID2];
      int edgeID2 = thinTri2.edgeID(maxedge);
      Real oldminangle = std::min(thinTri1.minAngle(), thinTri2.minAngle());

      YSB::Triangle<Real, 3> newTri1{thinTri1.vertex((maxedgeID + 2) % 3),
                                     thinTri1.vertex(maxedgeID),
                                     thinTri2.vertex((edgeID2 + 2) % 3)};
      YSB::Triangle<Real, 3> newTri2{thinTri2.vertex((edgeID2 + 2) % 3),
                                     thinTri2.vertex(edgeID2),
                                     thinTri1.vertex((maxedgeID + 2) % 3)};

      Real newminangle = std::min(newTri1.minAngle(), newTri2.minAngle());
      YSB::Segment<Real, 3> newDiag(thinTri1.vertex((maxedgeID + 2) % 3),
                                    thinTri2.vertex((edgeID2 + 2) % 3));
      if (newDiag.length() > chdLenRange.hi()[0] ||
          newDiag.length() < chdLenRange.lo()[0] || newminangle < minAngle)
        continue;

      // exclude the situation where the new triangle overlaps with the
      // original neighbor tiangle.
      auto neighborIDs1 = segNeighbor[i][newTri1.edge(0)];
      int neighborTriID1 =
          (neighborIDs1[0] == j ? neighborIDs1[1] : neighborIDs1[0]);
      std::vector<YSB::Segment<Real, 3>> tempresult1;
      vecVecTri[i][neighborTriID1].intersect(newTri1, tempresult1);
      if (tempresult1.size() > 2)
        continue;
      auto neighborIDs2 = segNeighbor[i][newTri1.edge(1)];
      int neighborTriID2 =
          (neighborIDs2[0] == triID2 ? neighborIDs2[1] : neighborIDs2[0]);
      std::vector<YSB::Segment<Real, 3>> tempresult2;
      vecVecTri[i][neighborTriID2].intersect(newTri1, tempresult2);
      if (tempresult2.size() > 2)
        continue;
      auto neighborIDs3 = segNeighbor[i][newTri2.edge(0)];
      int neighborTriID3 =
          (neighborIDs3[0] == triID2 ? neighborIDs3[1] : neighborIDs3[0]);
      std::vector<YSB::Segment<Real, 3>> tempresult3;
      vecVecTri[i][neighborTriID3].intersect(newTri2, tempresult3);
      if (tempresult3.size() > 2)
        continue;
      auto neighborIDs4 = segNeighbor[i][newTri2.edge(1)];
      int neighborTriID4 =
          (neighborIDs4[0] == j ? neighborIDs4[1] : neighborIDs4[0]);
      std::vector<YSB::Segment<Real, 3>> tempresult4;
      vecVecTri[i][neighborTriID4].intersect(newTri2, tempresult4);
      if (tempresult4.size() > 2)
        continue;

      vecVecTri[i][j] = newTri1;
      vecVecTri[i][triID2] = newTri2;

      segNeighbor[i].erase(maxedge);
      segNeighbor[i][newDiag] = {int(j), triID2};
      if (segNeighbor[i][newTri1.edge(1)][0] == triID2)
        segNeighbor[i][newTri1.edge(1)][0] = int(j);
      else
        segNeighbor[i][newTri1.edge(1)][1] = int(j);
      if (segNeighbor[i][newTri2.edge(1)][0] == int(j))
        segNeighbor[i][newTri2.edge(1)][0] = triID2;
      else
        segNeighbor[i][newTri2.edge(1)][1] = triID2;
    }
  }

  for (size_t i = 0; i < vecGSurf.size(); i++)
    vecGSurf[i] = YSB::GluedSurface<Real, 2>(std::move(vecVecTri[i]));
}

void EdgeSwapper::swapGlobally(YSB::YinSet<3, 2> &ys) {
  auto &vecGSurf = ys.boundaryRef().vecGluedSurfaceRef();
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
  std::vector<std::vector<YSB::Triangle<Real, 3>>> vecVecTri(vecGSurf.size());
  for (size_t i = 0; i < vecVecTri.size(); i++) {
    vecVecTri[i] = vecGSurf[i].vecTriangle();
  }

  for (size_t i = 0; i < vecGSurf.size(); i++) {
    for (int iter = 0; iter < 5; iter++) {
      bool flag = false;

      int count = 0;
      for (size_t k = 0; k < vecGSurf[i].vecTriangle().size(); k++) {
        const auto &tri = vecGSurf[i].vecTriangle()[k];
        if (tri.minAngle() < minAngle)
          count++;
      }

      for (size_t j = 0; j < vecGSurf[i].vecTriangle().size(); j++) {
        auto thinTri1 = vecVecTri[i][j];
        int maxedgeID = thinTri1.maxEdgeID();
        auto maxedge = thinTri1.edge(maxedgeID);
        assert(segNeighbor[i][maxedge].size() == 2 &&
               "Non-manifold edge exists!");

        auto triID2 = segNeighbor[i][maxedge][0] == j
                          ? segNeighbor[i][maxedge][1]
                          : segNeighbor[i][maxedge][0];
        auto thinTri2 = vecVecTri[i][triID2];
        int edgeID2 = thinTri2.edgeID(maxedge);
        Real oldminangle = std::min(thinTri1.minAngle(), thinTri2.minAngle());

        YSB::Triangle<Real, 3> newTri1{thinTri1.vertex((maxedgeID + 2) % 3),
                                       thinTri1.vertex(maxedgeID),
                                       thinTri2.vertex((edgeID2 + 2) % 3)};
        YSB::Triangle<Real, 3> newTri2{thinTri2.vertex((edgeID2 + 2) % 3),
                                       thinTri2.vertex(edgeID2),
                                       thinTri1.vertex((maxedgeID + 2) % 3)};

        Real newminangle = std::min(newTri1.minAngle(), newTri2.minAngle());
        YSB::Segment<Real, 3> newDiag(thinTri1.vertex((maxedgeID + 2) % 3),
                                      thinTri2.vertex((edgeID2 + 2) % 3));
        if (newDiag.length() > chdLenRange.hi()[0] ||
            newDiag.length() < chdLenRange.lo()[0] ||
            oldminangle > newminangle)
          continue;

        // exclude the situation where the new triangle overlaps with the
        // original neighbor tiangle.
        auto neighborIDs1 = segNeighbor[i][newTri1.edge(0)];
        int neighborTriID1 =
            (neighborIDs1[0] == j ? neighborIDs1[1] : neighborIDs1[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult1;
        vecVecTri[i][neighborTriID1].intersect(newTri1, tempresult1);
        if (tempresult1.size() > 2)
          continue;
        auto neighborIDs2 = segNeighbor[i][newTri1.edge(1)];
        int neighborTriID2 =
            (neighborIDs2[0] == triID2 ? neighborIDs2[1] : neighborIDs2[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult2;
        vecVecTri[i][neighborTriID2].intersect(newTri1, tempresult2);
        if (tempresult2.size() > 2)
          continue;
        auto neighborIDs3 = segNeighbor[i][newTri2.edge(0)];
        int neighborTriID3 =
            (neighborIDs3[0] == triID2 ? neighborIDs3[1] : neighborIDs3[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult3;
        vecVecTri[i][neighborTriID3].intersect(newTri2, tempresult3);
        if (tempresult3.size() > 2)
          continue;
        auto neighborIDs4 = segNeighbor[i][newTri2.edge(1)];
        int neighborTriID4 =
            (neighborIDs4[0] == j ? neighborIDs4[1] : neighborIDs4[0]);
        std::vector<YSB::Segment<Real, 3>> tempresult4;
        vecVecTri[i][neighborTriID4].intersect(newTri2, tempresult4);
        if (tempresult4.size() > 2)
          continue;

        flag = true;
        vecVecTri[i][j] = newTri1;
        vecVecTri[i][triID2] = newTri2;

        segNeighbor[i].erase(maxedge);
        segNeighbor[i][newDiag] = {int(j), triID2};
        if (segNeighbor[i][newTri1.edge(1)][0] == triID2)
          segNeighbor[i][newTri1.edge(1)][0] = int(j);
        else
          segNeighbor[i][newTri1.edge(1)][1] = int(j);
        if (segNeighbor[i][newTri2.edge(1)][0] == int(j))
          segNeighbor[i][newTri2.edge(1)][0] = triID2;
        else
          segNeighbor[i][newTri2.edge(1)][1] = triID2;
      }

      if (flag == false)
        break;
    }
  }

  for (size_t i = 0; i < vecGSurf.size(); i++)
    vecGSurf[i] = YSB::GluedSurface<Real, 2>(std::move(vecVecTri[i]));
}