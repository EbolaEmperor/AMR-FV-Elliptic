#include "AMRTools/Wrapper_Silo.h"

#include "Core/TensorExpr.h"

template <int Dim>
Wrapper_Silo<Dim>::Wrapper_Silo(std::string aFolder,
                                const std::string &aPrefix,
                                const AMRMeshHierachy<Dim> &amrHier,
                                int aCycle,
                                Real aTime,
                                int aRoot) :
    amrHier_(amrHier), root_(aRoot), time_(aTime), cycle_(aCycle) {
  // Check the filename
  simple_path thePath(aFolder);
  std::ostringstream prefix;
  prefix << aPrefix << '.' << std::setfill('0') << std::setw(4) << aCycle
         << ".proc" << std::setfill('0') << std::setw(3)
         << ProcID(MPI_COMM_WORLD);
  thePath.append(prefix.str() + ".silo");

  folder_ = aFolder;
  prefix_ = aPrefix;

  // Create the HDF5 file
  good_ = true;
  dbfile_ = nullptr;
  dbfile_ = DBCreate(thePath.c_str(), DB_CLOBBER, DB_LOCAL, nullptr, DB_HDF5);
  if (dbfile_ == nullptr) {
    good_ = false;
    return;
  }
  // Create the rectilinear meshes
  putAMRMesh();
}

template <int Dim>
void Wrapper_Silo<Dim>::close() {
  UnitTimer::getInstance().begin("Silo");
  if (dbfile_ != nullptr) {
    DBClose(dbfile_);
    dbfile_ = nullptr;
  }

  // Sync, then merge data files.
  MPI_Barrier(MPI_COMM_WORLD);
  if (ProcID(MPI_COMM_WORLD) == root_) {
    mergeAMRMesh();
    for (auto &var : vars_)
      mergeDatas(var.c_str());
    DBClose(dbfile_);
    dbfile_ = nullptr;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  UnitTimer::getInstance().end("Silo");
}

template <int Dim>
Wrapper_Silo<Dim>::~Wrapper_Silo() {
  close();
}

template <int Dim>
void Wrapper_Silo<Dim>::putAMRMesh() {
  UnitTimer::getInstance().begin("Silo");
  int proc_id = ProcID(MPI_COMM_WORLD);
  unsigned allbox = amrHier_.numBoxes(proc_id), pallbox = 0;

  char meshname[11];
  sprintf(meshname, "AMRMesh%03d", proc_id);
  char **submeshnames = new char *[allbox];
  int *submeshtype = new int[allbox];
  int hide = 1;

  DBoptlist *optlist = DBMakeOptlist(3);
  DBAddOption(optlist, DBOPT_DTIME, &time_);
  DBAddOption(optlist, DBOPT_CYCLE, &cycle_);
  DBAddOption(optlist, DBOPT_HIDE_FROM_GUI, &hide);

  for (unsigned level = 0; level < amrHier_.size(); ++level) {
    Real *pCoords[Dim];
    Tensor<Real, 1> coords[Dim];
    auto &domain = amrHier_.getDomain(level);
    auto &mesh = amrHier_.getMesh(level);
    const auto x0 = domain.getX0();
    const auto dx = domain.getDx();

    for (auto it = mesh.begin(); it.ok(); ++it) {
      int bid = it.index();
      if (mesh.getProcID(bid) != proc_id)
        continue;
      submeshtype[pallbox] = DB_QUAD_RECT;
      submeshnames[pallbox] = new char[12];
      sprintf(submeshnames[pallbox], "mesh_%02d_%03d", level, bid);

      const auto gridSize = it->size();
      const auto nodeSize = gridSize + 1;
      const auto gridLo = it->lo();

      for (int d = 0; d < Dim; ++d) {
        coords[d].resize(gridSize[d] + 1);
        pCoords[d] = coords[d].data();
        for (int i = 0; i <= gridSize[d]; ++i)
          coords[d](i) = x0[d] + (gridLo[d] + i) * dx[d];
      }

      assert(good_);
      if (DBPutQuadmesh(dbfile_,
                        submeshnames[pallbox],
                        nullptr,
                        pCoords,
                        nodeSize.data(),
                        Dim,
                        DB_DOUBLE,
                        DB_COLLINEAR,
                        optlist) != 0)
        good_ = false;

      ++pallbox;
    }
  }

  assert(good_);
  hide = 0;
  if (DBPutMultimesh(
          dbfile_, meshname, allbox, submeshnames, submeshtype, optlist) != 0)
    good_ = false;

  for (unsigned i = 0; i < allbox; ++i)
    delete[] submeshnames[i];
  delete[] submeshnames;
  delete[] submeshtype;

  DBFreeOptlist(optlist);
  UnitTimer::getInstance().end("Silo");
}

template <int Dim>
void Wrapper_Silo<Dim>::mergeAMRMesh() {
  UnitTimer::getInstance().begin("Silo");
  assert(good_);
  {
    simple_path thePath(folder_);
    std::ostringstream prefix;
    prefix << prefix_ << '.' << std::setfill('0') << std::setw(4) << cycle_;
    dbgcout1 << "Writing " << prefix.str() << std::endl;
    thePath.append(prefix.str() + ".silo");
    dbfile_ =
        DBCreate(thePath.c_str(), DB_CLOBBER, DB_LOCAL, nullptr, DB_HDF5);
    if (dbfile_ == nullptr) {
      good_ = false;
      return;
    }
  }

  int nProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  unsigned allbox = amrHier_.numBoxes(), pallbox = 0;

  char meshname[] = "AMRMesh";
  char **submeshnames = new char *[allbox];
  int *submeshtype = new int[allbox];

  std::vector<simple_path> thePath(nProcs);
  for (int i = 0; i < nProcs; ++i) {
    std::ostringstream prefix;
    prefix << prefix_ << '.' << std::setfill('0') << std::setw(4) << cycle_
           << ".proc" << std::setfill('0') << std::setw(3) << i;
    thePath[i].append(prefix.str() + ".silo");
  }

  for (unsigned level = 0; level < amrHier_.size(); ++level) {
    auto &mesh = amrHier_.getMesh(level);

    for (auto it = mesh.begin(); it.ok(); ++it) {
      int bid = it.index();
      int proc_id = mesh.getProcID(it.index());
      submeshtype[pallbox] = DB_QUAD_RECT;
      submeshnames[pallbox] = new char[thePath[proc_id].size() + 14];
      sprintf(submeshnames[pallbox],
              "%s:mesh_%02d_%03d",
              thePath[proc_id].c_str(),
              level,
              bid);
      ++pallbox;
    }
  }

  DBoptlist *optlist = DBMakeOptlist(2);
  DBAddOption(optlist, DBOPT_DTIME, &time_);
  DBAddOption(optlist, DBOPT_CYCLE, &cycle_);

  assert(good_);
  if (DBPutMultimesh(
          dbfile_, meshname, allbox, submeshnames, submeshtype, optlist) != 0)
    good_ = false;

  for (unsigned i = 0; i < allbox; ++i)
    delete[] submeshnames[i];
  delete[] submeshnames;
  delete[] submeshtype;

  DBFreeOptlist(optlist);
  UnitTimer::getInstance().end("Silo");
}

template <>
template <typename T>
void Wrapper_Silo<2>::putAMRScalar(const std::vector<LevelData<T, 2>> &aDatas,
                                   const char *varname,
                                   int comp) {
  UnitTimer::getInstance().begin("Silo");
  assert(good_);
  vars_.push_back(std::string(varname));

  int proc_id = ProcID(MPI_COMM_WORLD);
  unsigned len = strlen(varname);
  unsigned allsize = amrHier_.numBoxes(proc_id), pallsize = 0;

  char submeshname[12];
  char **subvarname = new char *[allsize];
  int *vartype = new int[allsize];
  int centering = aDatas[0].getCentering(comp), db_cent = 0;

  int db_type = -1;
  if (std::is_same<Real, T>::value)
    db_type = DB_DOUBLE;
  else if (std::is_same<int, T>::value)
    db_type = DB_INT;
  else if (std::is_same<bool, T>::value)
    db_type = DB_CHAR;
  assert(db_type != -1);

  if (centering == CellCenter)
    db_cent = DB_ZONECENT;
  else if (centering == NodeCenter)
    db_cent = DB_NODECENT;
  else {
    db_cent = DB_EDGECENT;
    throw std::runtime_error("Wrapper_Silo:: Scalar cannot be face-centered.");
  }

  int hide = 1;
  DBoptlist *varoptlist = DBMakeOptlist(3);
  DBAddOption(varoptlist, DBOPT_DTIME, &time_);
  DBAddOption(varoptlist, DBOPT_CYCLE, &cycle_);
  DBAddOption(varoptlist, DBOPT_HIDE_FROM_GUI, &hide);

  for (unsigned level = 0; level < amrHier_.size(); ++level) {
    const auto &aData = aDatas[level];

    for (auto it = aData.const_begin(); it.ok(); ++it) {
      int box_id = it.getBoxID(), varid = pallsize + it.index();
      subvarname[varid] = new char[len + 8];
      vartype[varid] = DB_QUADVAR;
      sprintf(submeshname, "mesh_%02d_%03d", level, box_id);
      sprintf(subvarname[varid], "%s_%02d_%03d", varname, level, box_id);

      auto box = it.getValidBox(comp);
      auto sz = box.size();
      Tensor<T, 2> pData = it.getData()[comp].slice(box);

      assert(good_);
      if (DBPutQuadvar1(dbfile_,
                        subvarname[varid],
                        submeshname,
                        pData.data(),
                        sz.data(),
                        2,
                        nullptr,
                        0,
                        db_type,
                        db_cent,
                        varoptlist) != 0)
        good_ = false;
    }

    pallsize += aData.size();
  }

  DBoptlist *optlist = DBMakeOptlist(3);
  char meshname[11];
  sprintf(meshname, "AMRMesh%03d", ProcID(MPI_COMM_WORLD));
  DBAddOption(optlist, DBOPT_DTIME, &time_);
  DBAddOption(optlist, DBOPT_CYCLE, &cycle_);
  DBAddOption(optlist, DBOPT_MMESH_NAME, meshname);

  assert(good_);
  if (DBPutMultivar(dbfile_, varname, allsize, subvarname, vartype, optlist) !=
      0)
    good_ = false;

  for (unsigned i = 0; i < allsize; ++i)
    delete[] subvarname[i];
  delete[] subvarname;
  delete[] vartype;

  DBFreeOptlist(varoptlist);
  DBFreeOptlist(optlist);
  UnitTimer::getInstance().end("Silo");
}

template <int Dim>
void Wrapper_Silo<Dim>::mergeDatas(const char *varname) {
  UnitTimer::getInstance().begin("Silo");
  assert(good_);

  int nProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  std::vector<simple_path> thePath(nProcs);
  for (int i = 0; i < nProcs; ++i) {
    std::ostringstream prefix;
    prefix << prefix_ << '.' << std::setfill('0') << std::setw(4) << cycle_
           << ".proc" << std::setfill('0') << std::setw(3) << i;
    thePath[i].append(prefix.str() + ".silo");
  }

  unsigned len = strlen(varname);
  unsigned allsize = amrHier_.numBoxes(), pallsize = 0;

  char submeshname[12];
  char **subvarname = new char *[allsize];
  int *vartype = new int[allsize];

  for (unsigned level = 0; level < amrHier_.size(); ++level) {
    const auto &mesh = amrHier_.getMesh(level);
    for (auto it = mesh.begin(); it.ok(); ++it) {
      int box_id = it.index(), varid = pallsize + it.index();
      int proc_id = mesh.getProcID(box_id);
      subvarname[varid] = new char[len + thePath[proc_id].size() + 10];
      vartype[varid] = DB_QUADVAR;
      sprintf(submeshname, "mesh_%02d_%03d", level, box_id);
      sprintf(subvarname[varid],
              "%s:%s_%02d_%03d",
              thePath[proc_id].c_str(),
              varname,
              level,
              box_id);
    }
    pallsize += mesh.size();
  }

  DBoptlist *optlist = DBMakeOptlist(3);
  char meshname[] = "AMRMesh";
  DBAddOption(optlist, DBOPT_DTIME, &time_);
  DBAddOption(optlist, DBOPT_CYCLE, &cycle_);
  DBAddOption(optlist, DBOPT_MMESH_NAME, meshname);

  assert(good_);
  if (DBPutMultivar(dbfile_, varname, allsize, subvarname, vartype, optlist) !=
      0)
    good_ = false;

  for (unsigned i = 0; i < allsize; ++i)
    delete[] subvarname[i];
  delete[] subvarname;
  delete[] vartype;

  DBFreeOptlist(optlist);
  UnitTimer::getInstance().end("Silo");
}

template class Wrapper_Silo<2>;

template void Wrapper_Silo<2>::putAMRScalar(
    const std::vector<LevelData<Real, 2>> &,
    const char *,
    int);

template void Wrapper_Silo<2>::putAMRScalar(
    const std::vector<LevelData<int, 2>> &,
    const char *,
    int);

template void Wrapper_Silo<2>::putAMRScalar(
    const std::vector<LevelData<bool, 2>> &,
    const char *,
    int);