#ifndef ROWSPARSE_H
#define ROWSPARSE_H

#include "Core/MPI.h"
#include "Core/Tensor.h"

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <map>
#include <math.h>
#include <ostream>
#include <stdexcept>
#include <vector>

template <class T_RowIndex = int, class T_ColIndex = T_RowIndex>
class RowSparse {
public:
  RowSparse() = default;

  std::size_t insertRow(const T_RowIndex &r,
                        std::initializer_list<T_ColIndex> cols,
                        std::initializer_list<Real> vals);

  std::size_t insertRow(const T_RowIndex &r,
                        int nz,
                        const T_ColIndex *cols,
                        const Real *vals);

  std::size_t getNumRows() const { return rows.size(); }

  // calculate z = alpha * Ax + y
  template <class T1, class T2, class T3>
  void AXPY(Real alpha, const T1 &x, const T2 &y, T3 &z) const;

  // calculate z = alpha * ax + y, compressed row output
  template <class T1, class T2>
  Tensor<Real, 1> AXPY(Real alpha, const T1 &x, const T2 &y) const;

  void clear() {
    rows.clear();
    nonZeros.clear();
    columns.clear();
    values.clear();
  }

  // iterators
public:
  class RowSparseIterator {
    typename std::vector<Real>::const_iterator pVal;
    typename std::vector<T_ColIndex>::const_iterator pCol;
    typename std::vector<int>::const_iterator pNz;
    typename std::vector<T_RowIndex>::const_iterator pRw;

  public:
    friend class RowSparse;
    const Real &getValue(int j) const { return *(pVal + j); }
    const T_ColIndex &getColumn(int j) const { return *(pCol + j); }
    const T_RowIndex &getRow() const { return *pRw; }
    int getNz() const { return *pNz; }

    RowSparseIterator &operator++() {
      pVal += getNz();
      pCol += getNz();
      ++pNz;
      ++pRw;
      return *this;
    }
    void operator++(int) { ++(*this); }
    bool operator!=(const RowSparseIterator &rhs) const {
      return this->pRw != rhs.pRw;
    }
    bool operator==(const RowSparseIterator &rhs) const {
      return this->pRw == rhs.pRw;
    }
    std::ptrdiff_t operator-(const RowSparseIterator &rhs) const {
      return this->pRw - rhs.pRw;
    }

    friend RowSparse<T_RowIndex, T_ColIndex> mergeRowSparse(
        const std::vector<RowSparse<T_RowIndex, T_ColIndex>> &vrs) {
      RowSparse<T_RowIndex, T_ColIndex> rs_new;
      for (auto it = vrs.cbegin(); it != vrs.cend(); ++it) {
        const auto &rs = *it;
        for (auto rit = rs.cbegin(); rit != rs.cend(); ++rit) {
          auto *pCol = &(*rit.pCol);
          auto *pVal = &(*rit.pVal);
          rs_new.insertRow(rit.getRow(), rit.getNz(), pCol, pVal);
        }
      }
      return rs_new;
    }
  };

  RowSparseIterator cbegin() const {
    RowSparseIterator it;
    it.pVal = values.cbegin();
    it.pCol = columns.cbegin();
    it.pNz = nonZeros.cbegin();
    it.pRw = rows.cbegin();
    return it;
  }

  RowSparseIterator cend() const {
    RowSparseIterator it;
    it.pRw = rows.cend();
    return it;
  }

  template <class T_Less = std::less<T_RowIndex>>
  RowSparseIterator find_fast(const T_RowIndex &ri, const T_Less &ls) const;

  template <class T_Equal = std::less<T_RowIndex>>
  RowSparseIterator find(const T_RowIndex &ri, const T_Equal &ls) const;

protected:
  std::vector<Real> values;
  std::vector<T_ColIndex> columns;
  std::vector<int> nonZeros;
  std::vector<int> rowBegin;
  std::vector<T_RowIndex> rows;

public:
  template <class RowCompare, class ColCompare>
  void sort(const RowCompare &, const ColCompare &);

  friend RowSparse<T_RowIndex, T_ColIndex> mergeRowSparse(
      const std::vector<RowSparse<T_RowIndex, T_ColIndex>> &vrs);

  friend RowSparse<T_RowIndex, T_ColIndex> allreduceRS(
      const RowSparse<T_RowIndex, T_ColIndex> &rs,
      MPI_Comm comm);

  friend std::ostream &operator<<(std::ostream &os, const RowSparse &rs) {
    for (auto it = rs.cbegin(); it != rs.cend(); ++it) {
      os << "row: " << it.getRow();
      int nz = it.getNz();
      for (int i = 0; i < nz; ++i) {
        os << "(" << it.getColumn(i) << "," << it.getValue(i) << ")";
      }
      os << "\n";
    }
    return os;
  }
};
//============================================================

template <class T_RowIndex, class T_ColIndex>
inline std::size_t RowSparse<T_RowIndex, T_ColIndex>::insertRow(
    const T_RowIndex &r,
    int nz,
    const T_ColIndex *cols,
    const Real *vals) {
  std::size_t a = rows.size();
  rows.push_back(r);
  rowBegin.push_back(values.size());
  nonZeros.push_back(nz);
  values.insert(values.cend(), vals, vals + nz);
  columns.insert(columns.cend(), cols, cols + nz);
  return a;
}

template <class T_RowIndex, class T_ColIndex>
inline std::size_t RowSparse<T_RowIndex, T_ColIndex>::insertRow(
    const T_RowIndex &r,
    std::initializer_list<T_ColIndex> cols,
    std::initializer_list<Real> vals) {
  return insertRow(r, cols.size(), cols.begin(), vals.begin());
}

template <class T_RowIndex, class T_ColIndex>
template <class T1, class T2, class T3>
inline void RowSparse<T_RowIndex, T_ColIndex>::AXPY(Real alpha,
                                                    const T1 &x,
                                                    const T2 &y,
                                                    T3 &z) const {
  for (auto rowIt = cbegin(); rowIt != cend(); ++rowIt) {
    Real Ax = 0.0;
    for (int k = 0; k < rowIt.getNz(); ++k)
      Ax += rowIt.getValue(k) * x(rowIt.getColumn(k));
    z(rowIt.getRow()) = alpha * Ax + y(rowIt.getRow());
  }
}

template <class T_RowIndex, class T_ColIndex>
template <class T1, class T2>
inline Tensor<Real, 1> RowSparse<T_RowIndex, T_ColIndex>::AXPY(
    Real alpha,
    const T1 &x,
    const T2 &y) const {
  Tensor<Real, 1> axpy(getNumRows());
  int i = 0;
  for (auto rowIt = cbegin(); rowIt != cend(); ++rowIt, ++i) {
    Real Ax = 0.0;
    for (int k = 0; k < rowIt.getNz(); ++k) {
      auto first = rowIt.getValue(k);
      auto second = x(rowIt.getColumn(k));
      Ax += first * second;
      if (isnan(Ax)) {
        std::cerr << "AXPY: k = " << k << ", column = " << rowIt.getColumn(k)
                  << std::endl;
        throw std::runtime_error("error: Ax is nan!");
      }
    }
    auto y_value = y(rowIt.getRow());
    axpy(i) = alpha * Ax + y_value;
  }
  return axpy;
}

template <class T_RowIndex, class T_ColIndex>
template <class T_Less>
inline auto RowSparse<T_RowIndex, T_ColIndex>::find_fast(
    const T_RowIndex &ri,
    const T_Less &ls) const -> RowSparseIterator {
  auto it = std::lower_bound(rows.cbegin(), rows.cend(), ri, ls);
  if (it == rows.cend() || ls(ri, *it))
    return cend();
  auto i = it - rows.cbegin();
  RowSparseIterator rsit;
  rsit.pRw = it;
  rsit.pNz = nonZeros.cbegin() + i;
  rsit.pCol = columns.cbegin() + rowBegin[i];
  rsit.pVal = values.cbegin() + rowBegin[i];
  return rsit;
}

template <class T_RowIndex, class T_ColIndex>
template <class T_Equal>
inline auto RowSparse<T_RowIndex, T_ColIndex>::find(const T_RowIndex &ri,
                                                    const T_Equal &ls) const
    -> RowSparseIterator {
  auto equal = [&](const T_RowIndex x1) { return ls.compare(x1, ri) == 0; };
  auto it = std::find_if(rows.cbegin(), rows.cend(), equal);
  if (it == rows.cend() || ls.compare(ri, *it) != 0)
    return cend();
  auto i = it - rows.cbegin();
  RowSparseIterator rsit;
  rsit.pRw = it;
  rsit.pNz = nonZeros.cbegin() + i;
  rsit.pCol = columns.cbegin() + rowBegin[i];
  rsit.pVal = values.cbegin() + rowBegin[i];
  return rsit;
}

template <class T_RowIndex, class T_ColIndex>
template <class RowCompare, class ColCompare>
void RowSparse<T_RowIndex, T_ColIndex>::sort(const RowCompare &rowcompare,
                                             const ColCompare &colcompare) {
  // init
  std::map<T_RowIndex, int, RowCompare> rowIdx;
  for (size_t i = 0; i != rows.size(); ++i) {
    rowIdx[rows[i]] = i;
  }
  std::sort(rows.begin(), rows.end(), rowcompare);
  // reset other data.
  std::vector<Real> tempValues;
  std::vector<T_ColIndex> tempColumns;
  std::vector<int> tempNonZeros;
  std::vector<int> tempRowBegin;
  int pole = 0;
  for (size_t i = 0; i != rows.size(); ++i) {
    // get the row information
    int idx = rowIdx[rows[i]];
    int nz = nonZeros[idx];
    int start = rowBegin[idx];
    tempRowBegin.push_back(pole);
    tempNonZeros.push_back(nz);
    // sort the column
    std::vector<Real> rowvalues(nz);
    std::vector<T_ColIndex> rowcols(nz);
    std::copy(
        columns.data() + start, columns.data() + start + nz, rowcols.data());
    std::map<T_ColIndex, int, ColCompare> colIdx;
    for (int j = 0; j != nz; ++j) {
      colIdx[rowcols[j]] = j;
    }
    std::sort(rowcols.begin(), rowcols.end(), colcompare);
    for (int j = 0; j != nz; ++j) {
      rowvalues[j] = values[start + colIdx[rowcols[j]]];
    }
    for (int j = 0; j < nz; ++j) {
      tempValues.push_back(rowvalues[j]);
      tempColumns.push_back(rowcols[j]);
    }
    pole += nz;
  }
  values.swap(tempValues);
  columns.swap(tempColumns);
  nonZeros.swap(tempNonZeros);
  rowBegin.swap(tempRowBegin);
};

#endif  // ROWSPARSE_H
