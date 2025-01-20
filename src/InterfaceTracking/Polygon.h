#ifndef INTERFACETACKING_POLYGON_H
#define INTERFACETACKING_POLYGON_H
#include "YinSet/PointsLocater.h"
#include "YinSet3D/Triangle.h"

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <fstream>
#include <map>
#include <random>
#include <sys/stat.h>
#include <utility>

class Polygon {
public:
  // for CGAL use
  using K = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Itag = CGAL::No_intersection_tag;
  // using Itag =  CGAL::Exact_predicates_tag;
  using CDT =
      CGAL::Constrained_Delaunay_triangulation_2<K, CGAL::Default, Itag>;
  using CPoint = CDT::Point;
  using CPolygon = CGAL::Polygon_2<K>;
  using CFacehandle = CDT::Face_handle;

  using rVec = Vec<Real, 2>;
  Polygon(const std::vector<Segment<2>> &avecE) {
    assert(avecE.size() > 2 && "It is not a polygon");
    Real tol = YSB::Tolerance::Instance()->getTol();
    VecCompare<Real, 2> vCmp(tol);

    std::map<rVec, std::vector<Segment<2>>, VecCompare<Real, 2>> neighbor;
    for (auto &&e : avecE) {
      neighbor[e.p[0]].push_back(e);
      neighbor[e.p[1]].push_back(e);
    }
    for (auto &&an : neighbor) {
      assert(an.second.size() == 2 && "It is not a polygon");
      vecP.push_back(an.first);
    }
    Segment<2> tempE = avecE[0];
    vecE.push_back(tempE);
    for (size_t i = 1; i < avecE.size(); i++) {
      auto pn = neighbor[tempE.p[1]];
      if (vCmp.compare(pn[0].p[0], tempE.p[0]) == 0) {
        // if(vCmp.compare(pn[1].p[0], tempE.p[1]) != 0)
        // {
        //     exportMatlab(avecE, "wrongpg",
        //     "/home/qiuyunhao/GitHub/splineMARS/code/Yinsets3D_boolean/res/MARS3D/pg/");
        // }
        assert(vCmp.compare(pn[1].p[0], tempE.p[1]) == 0 &&
               "The orientation of the edge is wrong");
        tempE = pn[1];
      } else {
        // if(vCmp.compare(pn[0].p[0], tempE.p[1]) != 0)
        // {
        //     exportMatlab(avecE, "wrongpg",
        //     "/home/qiuyunhao/GitHub/splineMARS/code/Yinsets3D_boolean/res/MARS3D/pg/");
        // }
        assert(vCmp.compare(pn[0].p[0], tempE.p[1]) == 0 &&
               "The orientation of the edge is wrong");
        tempE = pn[0];
      }
      vecE.push_back(tempE);
    }

    // exportMatlab(avecE, "asd",
    // "/home/qiuyunhao/GitHub/splineMARS/code/Yinsets3D_boolean/res/MARS3D/pg/");

    Real area = 0;
    for (auto &&e : vecE) {
      area += 0.5 * (e.p[0][0] + e.p[1][0]) * (e.p[1][1] - e.p[0][1]);
    }
    if (area == abs(area))
      bounded = 1;
    else
      bounded = 0;
  }
  std::vector<rVec> generateInnerPoint(int n) const;
  void correctOrientation(std::vector<YSB::Triangle<Real, 2>> &vecTri) const;
  std::vector<YSB::Triangle<Real, 2>> triangulate(
      const std::vector<rVec> &pts) const;
  Real Area() const;

  void exportMatlab(std::vector<Segment<2>> edges,
                    const std::string &name,
                    const std::string &folder,
                    int precision = 6) const {
    const char *path = folder.c_str();
    int isCreate =
        mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    if (!isCreate)
      printf("create path:%s\n", path);
    // else
    //   printf("create path failed! error code : %d %s \n", isCreate, path);
    std::ofstream outfile(folder + "/" + name + ".m");
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(precision);
    for (int i = 0; i < 2; i++) {
      outfile << "x"
              << "_" << i << " = [";
      for (size_t j = 0; j < edges.size(); j++) {
        outfile << edges[j].p[0][i] << ", ";
      }
      outfile << edges[0].p[0][i] << "];" << std::endl;
    }
    outfile << "plot(x"
            << "_0, x"
            << "_1, 'k');" << std::endl;
    outfile << "hold on;" << std::endl << std::endl;
  }
  void exportMatlab(const std::vector<YSB::Triangle<Real, 2>> &vecTri,
                    const std::string &name,
                    const std::string &folder,
                    int precision = 6) const {
    const char *path = folder.c_str();
    int isCreate =
        mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    if (!isCreate)
      printf("create path:%s\n", path);
    // else
    //   printf("create path failed! error code : %d %s \n", isCreate, path);
    std::ofstream outfile(folder + "/" + name + ".m");
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(precision);
    for (size_t j = 0; j < vecTri.size(); j++) {
      auto tri = vecTri[j];
      for (int i = 0; i < 2; i++) {
        outfile << "x" << j << "_" << i << " = ";
        outfile << "[" << tri.vertex(0)[i] << ", " << tri.vertex(1)[i] << ", "
                << tri.vertex(2)[i] << ", " << tri.vertex(0)[i] << "];"
                << std::endl;
      }
      outfile << "plot(x" << j << "_0, x" << j << "_1, 'k');" << std::endl;
      outfile << "hold on;" << std::endl << std::endl;
    }
  }
  const std::vector<Segment<2>> &getEdge() const { return vecE; }
  const std::vector<rVec> &getVertex() const { return vecP; }

private:
  std::vector<Segment<2>> vecE;
  std::vector<rVec> vecP;
  int bounded;  // 1 bounded, 0 unbounded
};

#endif