
#include "Core/RowSparse.h"

#include "Core/Config.h"
#include "Core/MPI.h"
#include "Core/type_traits.h"
#include "mpi.h"

#include <numeric>

// TODO there are bugs in this function!
// RowSparse<Vec<int, 2>, Vec<int, 2>>
// allreduceRS(const RowSparse<Vec<int, 2>, Vec<int, 2>> &rs, MPI_Comm comm) {
//   RowSparse<Vec<int, 2>, Vec<int, 2>> result;
//   int num_procs = numProcs(comm);
//   int proc_id = ProcID(comm);
//   // first get number of values and rows
//   std::vector<int> num_values_vector(num_procs);
//   std::vector<int> num_rows_vector(num_procs);
//   num_values_vector.at(proc_id) = rs.values.size();
//   num_rows_vector.at(proc_id) = rs.rows.size();
//   MPI_Allreduce(MPI_IN_PLACE, num_values_vector.data(), num_procs, MPI_INT,
//                 MPI_SUM, comm);
//   MPI_Allreduce(MPI_IN_PLACE, num_rows_vector.data(), num_procs, MPI_INT,
//                 MPI_SUM, comm);
//   int num_values =
//       std::accumulate(num_values_vector.begin(), num_values_vector.end(),
//       0);
//   int num_rows =
//       std::accumulate(num_rows_vector.begin(), num_rows_vector.end(), 0);
//   int num_values_front = std::accumulate(
//       num_values_vector.begin(), num_values_vector.begin() + proc_id, 0);
//   int num_rows_front = std::accumulate(num_rows_vector.begin(),
//                                        num_rows_vector.begin() + proc_id,
//                                        0);
//   // construct the result
//   result.values.resize(num_values);
//   result.columns.resize(num_values);
//   std::vector<int> columns1(num_values);
//   std::vector<int> columns2(num_values);
//   for (int i = num_values_front; (uint)i < num_rows_front +
//   rs.values.size();
//        ++i) {
//     columns1.at(i) = rs.columns[i][0];
//     columns2.at(i) = rs.columns[i][1];
//   }
//   result.nonZeros.resize(num_rows);
//   result.rowBegin.resize(num_rows);
//   result.rows.resize(num_rows);
//   std::vector<int> rows1(num_rows);
//   std::vector<int> rows2(num_rows);
//   for (int i = num_rows_front; (uint)i < num_rows_front + rs.rows.size();
//   ++i) {
//     rows1.at(i) = rs.rows[i][0];
//     rows2.at(i) = rs.rows[i][1];
//   }
//   std::vector<int> values_displs(num_procs, 0);
//   std::vector<int> rows_displs(num_procs, 0);
//   int svalues = 0, srows = 0;
//   for (int i = 1; i < num_procs; ++i) {
//     values_displs.at(i) = svalues;
//     rows_displs.at(i) = srows;
//     svalues += num_values_vector.at(i - 1);
//     srows += num_rows_vector.at(i - 1);
//   }
//   // now get the values and rows
//   MPI_Allgatherv(rs.values.data(), rs.values.size(), MPI_DOUBLE,
//                  result.values.data(), num_values_vector.data(),
//                  values_displs.data(), MPI_DOUBLE, comm);
//   MPI_Allreduce(MPI_IN_PLACE, columns1.data(), num_values, MPI_INT, MPI_SUM,
//                 comm);
//   MPI_Allreduce(MPI_IN_PLACE, columns2.data(), num_values, MPI_INT, MPI_SUM,
//                 comm);
//   MPI_Allgatherv(rs.nonZeros.data(), rs.nonZeros.size(), MPI_INT,
//                  result.nonZeros.data(), num_rows_vector.data(),
//                  rows_displs.data(), MPI_INT, comm);
//   MPI_Allgatherv(rs.rowBegin.data(), rs.rowBegin.size(), MPI_INT,
//                  result.rowBegin.data(), num_rows_vector.data(),
//                  rows_displs.data(), MPI_INT, comm);
//   MPI_Allreduce(MPI_IN_PLACE, rows1.data(), num_rows, MPI_INT, MPI_SUM,
//   comm); MPI_Allreduce(MPI_IN_PLACE, rows2.data(), num_rows, MPI_INT,
//   MPI_SUM, comm); for (int i = 0; i < num_values; ++i) {
//     result.columns[i] = {columns1.at(i), columns2.at(i)};
//   }
//   for (int i = 0; i < num_rows; ++i) {
//     result.rows[i] = {rows1.at(i), rows2.at(i)};
//   }
//   return result;
// }

RowSparse<int, int> allreduceRS(const RowSparse<int, int> &rs, MPI_Comm comm) {
  RowSparse<int, int> result;
  int num_procs = numProcs(comm);
  int proc_id = ProcID(comm);
  // first get number of values and rows
  std::vector<int> num_values_vector(num_procs);
  std::vector<int> num_rows_vector(num_procs);
  num_values_vector.at(proc_id) = rs.values.size();
  num_rows_vector.at(proc_id) = rs.rows.size();
  MPI_Allreduce(MPI_IN_PLACE,
                num_values_vector.data(),
                num_procs,
                MPI_INT,
                MPI_SUM,
                comm);
  MPI_Allreduce(
      MPI_IN_PLACE, num_rows_vector.data(), num_procs, MPI_INT, MPI_SUM, comm);
  int num_values =
      std::accumulate(num_values_vector.begin(), num_values_vector.end(), 0);
  int num_rows =
      std::accumulate(num_rows_vector.begin(), num_rows_vector.end(), 0);
  int num_values_front = std::accumulate(
      num_values_vector.begin(), num_values_vector.begin() + proc_id, 0);
  int num_rows_front = std::accumulate(
      num_rows_vector.begin(), num_rows_vector.begin() + proc_id, 0);
  // construct the result
  result.values.resize(num_values);
  result.columns.resize(num_values);
  for (int i = num_values_front; (uint)i < num_values_front + rs.values.size();
       ++i) {
    result.columns.at(i) = rs.columns[i - num_values_front];
    result.values.at(i) = rs.values[i - num_values_front];
  }
  result.nonZeros.resize(num_rows);
  result.rowBegin.resize(num_rows);
  result.rows.resize(num_rows);
  for (int i = num_rows_front; (uint)i < num_rows_front + rs.rows.size();
       ++i) {
    result.rows.at(i) = rs.rows[i - num_rows_front];
    result.nonZeros.at(i) = rs.nonZeros[i - num_rows_front];
    result.rowBegin.at(i) = num_values_front + rs.rowBegin[i - num_rows_front];
  }
  // now get the values and rows
  MPI_Allreduce(MPI_IN_PLACE,
                result.values.data(),
                num_values,
                MPI_DOUBLE,
                MPI_SUM,
                comm);
  MPI_Allreduce(
      MPI_IN_PLACE, result.columns.data(), num_values, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(
      MPI_IN_PLACE, result.nonZeros.data(), num_rows, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(
      MPI_IN_PLACE, result.rows.data(), num_rows, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(
      MPI_IN_PLACE, result.rowBegin.data(), num_rows, MPI_INT, MPI_SUM, comm);
  return result;
}

template class RowSparse<int, int>;