/**
 * @file Wrapper_Silo.h
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 */

#pragma once

#include <AMRTools/AMRMeshHierachy.h>
#include <AMRTools/LevelData.h>
#include <AMRTools/Utilities.h>
#include <silo.h>
#include <vector>

template <int Dim>
class Wrapper_Silo {
public:
  using iVec = Vec<int, Dim>;

  Wrapper_Silo(std::string aFolder,
               const std::string &aPrefix,
               const AMRMeshHierachy<Dim> &amrHier,
               int aCycle = 0,
               Real aTime = 0.0,
               int root = 0);

  ~Wrapper_Silo();

public:
  template <typename T>
  void putAMRScalar(const std::vector<LevelData<T, Dim>> &aDatas,
                    const char *varname,
                    int comp = 0);

  operator bool() const { return good_; }

  void close();

protected:
  void putAMRMesh();

  void mergeAMRMesh();

  void mergeDatas(const char *varname);

protected:
  std::string folder_;
  std::string prefix_;

  DBfile *dbfile_;
  bool good_;
  const AMRMeshHierachy<Dim> &amrHier_;
  int root_;
  int centering_;
  Real time_;
  int cycle_;

  std::vector<std::string> vars_;
};