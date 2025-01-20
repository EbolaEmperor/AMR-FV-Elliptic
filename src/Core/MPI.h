#pragma once

#include "Core/Tensor.h"

#include <climits>
#include <limits>
#include <mpi.h>
#include <stdint.h>

using Communicator = MPI_Comm;
static constexpr int MPI_TAG = std::numeric_limits<int>::max();

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

inline int ProcID(Communicator comm = MPI_COMM_WORLD) {
  int id;
  MPI_Comm_rank(comm, &id);
  return id;
}

inline int numProcs(Communicator comm = MPI_COMM_WORLD) {
  int num;
  MPI_Comm_size(comm, &num);
  return num;
}

template <int Dim>
inline void sendTensor(const Tensor<Real, Dim> &buffer,
                       int dest,
                       MPI_Request &request,
                       int tag = MPI_TAG,
                       Communicator comm = MPI_COMM_WORLD) {
  MPI_Isend(
      buffer.data(), buffer.volume(), MPI_DOUBLE, dest, tag, comm, &request);
}

template <int Dim>
inline void sendTensor(const Tensor<int, Dim> &buffer,
                       int dest,
                       MPI_Request &request,
                       int tag = MPI_TAG,
                       Communicator comm = MPI_COMM_WORLD) {
  MPI_Isend(
      buffer.data(), buffer.volume(), MPI_INT, dest, tag, comm, &request);
}

template <int Dim>
inline void receiveTensor(Tensor<Real, Dim> &buffer,
                          int source,
                          MPI_Request &request,
                          int tag = MPI_TAG,
                          Communicator comm = MPI_COMM_WORLD) {
  MPI_Irecv(
      buffer.data(), buffer.volume(), MPI_DOUBLE, source, tag, comm, &request);
}

template <int Dim>
inline void receiveTensor(Tensor<int, Dim> &buffer,
                          int source,
                          MPI_Request &request,
                          int tag = MPI_TAG,
                          Communicator comm = MPI_COMM_WORLD) {
  MPI_Irecv(
      buffer.data(), buffer.volume(), MPI_INT, source, tag, comm, &request);
}

template <int Dim>
inline void reduceTensor(Tensor<Real, Dim> &out,
                         const Tensor<Real, Dim> &in,
                         MPI_Op op,
                         int root = 0,
                         Communicator comm = MPI_COMM_WORLD) {
  if (ProcID(comm) == root)
    assert(out.volume() == in.volume());
  MPI_Reduce(in.data(), out.data(), in.volume(), MPI_DOUBLE, op, root, comm);
}

template <int Dim>
inline void allReduceTensor(Tensor<Real, Dim> &out,
                            const Tensor<Real, Dim> &in,
                            MPI_Op op,
                            Communicator comm = MPI_COMM_WORLD) {
  MPI_Allreduce(in.data(), out.data(), in.volume(), MPI_DOUBLE, op, comm);
}

template <int Dim>
inline void bcastTensor(Tensor<Real, Dim> &buffer,
                        int root = 0,
                        Communicator comm = MPI_COMM_WORLD) {
  MPI_Bcast(buffer.data(), buffer.volume(), MPI_DOUBLE, root, comm);
}

template <typename T>
inline void mpiPrint(const T &rhs,
                     std::string name = "",
                     int root = 0,
                     Communicator comm = MPI_COMM_WORLD) {
  if (ProcID(comm) == root) {
    std::cout << name << std::endl;
    std::cout << rhs << std::endl;
  }
}

#define mpiStopAll(stop)                                                      \
  bool stop = true;                                                           \
  while (stop) {                                                              \
    sleep(2);                                                                 \
  }

#define mpiStop0(stop0)                                                       \
  bool stop0 = true;                                                          \
  while (stop0 && ProcID() == 0) {                                            \
    sleep(2);                                                                 \
  }
#define mpiStop1(stop1)                                                       \
  bool stop1 = true;                                                          \
  while (stop1 && ProcID() == 1) {                                            \
    sleep(2);                                                                 \
  }
#define mpiStop2(stop2)                                                       \
  bool stop2 = true;                                                          \
  while (stop2 && ProcID() == 2) {                                            \
    sleep(2);                                                                 \
  }
#define mpiStop3(stop3)                                                       \
  bool stop3 = true;                                                          \
  while (stop3 && ProcID() == 3) {                                            \
    sleep(2);                                                                 \
  }

#define mpicout (ProcID() == 0) && std::cout
#define dbgcout (get_dbglevel() >= 0) && std::cout
#define dbgcout1 (get_dbglevel() >= 1) && std::cout << "  "
#define dbgcout2 (get_dbglevel() >= 2) && std::cout << "    "
#define dbgcout3 (get_dbglevel() >= 3) && std::cout << "      "