/**
 * @file LevelGhostBoxes.cpp
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 * @copyright Copyright (c) 2024 Wenchong Huang
 *
 */
#include "AMRTools/LevelGhostBoxes.h"

template <int Dim>
void LevelGhostBoxes<Dim>::initLevelGhostBoxes() {
  // First, find ghosts located in the mesh
  for (auto it = mesh_.begin(); it.ok(); ++it) {
    auto ghostBox = it->grow(nGhost_);
    for (auto jt = mesh_.begin(); jt.ok(); ++jt) {
      if (it == jt)
        continue;
      Box<Dim> intersection = ghostBox & (*jt);
      if (!intersection.empty()) {
        ghostBoxes_.push_back(std::move(intersection));
        belongs_.push_back(it.index());
        location_.push_back(jt.index());
        bdryOfWhichDomain_.push_back(std::nullopt);
        bdryOfWhichFace_.push_back(std::nullopt);
      }
    }
  }

  // Second, find ghosts located out of the mesh

  const auto &pdly = domain_.getLayout();

  // Loop all faces of all boxes.
  for (auto it = mesh_.begin(); it.ok(); ++it) {
    // Extract not-intergrid ghost boxes in a side of a box.
    auto extractBdry = [&](const Box<Dim> &boxSide,
                           int cent,
                           bool isLowSide,
                           std::optional<int> bdryDomain = std::nullopt,
                           std::optional<int> bdryFace = std::nullopt) {
      iVec pre = boxSide.lo();
      loop_box_2(boxSide, i0, i1) {
        iVec idx = {i0, i1}, jdx = idx;
        bool intergrid = (mesh_.whichBox(idx).has_value());
        bool addnow = false;
        if (intergrid && idx[cent ^ 1] - pre[cent ^ 1] >= 1) {
          jdx[cent ^ 1]--;
          addnow = true;
        }
        if (!intergrid && idx == boxSide.hi())
          addnow = true;

        // Add a ghost box from pre to idx, with bandwidth nGhost
        if (addnow) {
          if (isLowSide) {
            iVec ipre = pre;
            ipre[cent] -= nGhost_ - 1;
            ghostBoxes_.emplace_back(ipre, jdx);
          } else {
            jdx[cent] += nGhost_ - 1;
            ghostBoxes_.emplace_back(pre, jdx);
          }
          belongs_.push_back(it.index());
          location_.push_back(std::nullopt);
          bdryOfWhichDomain_.push_back(bdryDomain);
          bdryOfWhichFace_.push_back(bdryFace);
        }

        if (intergrid)
          pre = idx, ++pre[cent ^ 1];
      }
    };

    int domainID = pdly.whichBox(it->lo()).value();
    const auto &domainBox = pdly.getBox(domainID);

    // left and down faces
    for (unsigned cent = 0; cent < Dim; ++cent) {
      auto domainBdryBox = domainBox.lowSideBox(cent);
      std::optional<int> bdryDomainID = std::nullopt;
      std::optional<int> bdryFaceID = cent << 1;
      if (!domain_.isInnerFace(domainID, bdryFaceID.value()) &&
          domainBdryBox.contain(it->lowSideBox(cent))) {
        bdryDomainID = domainID;
      }
      iVec sft = {0, 0};
      sft[cent] = -1;
      extractBdry(it->lowSideBox(cent).shift(sft),
                  cent,
                  true,
                  bdryDomainID,
                  bdryFaceID);
    }

    // right and up faces
    for (unsigned cent = 0; cent < Dim; ++cent) {
      auto domainBdryBox = domainBox.highSideBox(cent);
      std::optional<int> bdryDomainID = std::nullopt;
      std::optional<int> bdryFaceID = cent << 1 | 1;
      if (!domain_.isInnerFace(domainID, bdryFaceID.value()) &&
          domainBdryBox.contain(it->highSideBox(cent))) {
        bdryDomainID = domainID;
      }
      iVec sft = {0, 0};
      sft[cent] = 1;
      extractBdry(it->highSideBox(cent).shift(sft),
                  cent,
                  false,
                  bdryDomainID,
                  bdryFaceID);
    }
  }
}

template class LevelGhostBoxes<2>;