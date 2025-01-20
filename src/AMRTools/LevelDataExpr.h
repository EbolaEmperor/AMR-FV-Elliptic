/**
 * @file LevelDataExpr.h
 * @author {JiatuYan} ({2513630371@qq.com})
 * @brief Template Expression for LevelData
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <AMRTools/LevelData.h>
#include <AMRTools/Utilities.h>
#include <Core/numlib.h>
#include <functional>
#include <type_traits>

template <class T, int Dim>
template <class LDExpr>
inline LevelData<T, Dim> &LevelData<T, Dim>::operator=(
    const LevelDataExpr<LDExpr> &expr) {
  UnitTimer::getInstance().begin("LevelDataExpr");
  assert(expr.getMesh() == mesh_);
  assert(expr.getnComps() == nComps_);
  for (auto res_itr = begin(); res_itr.ok(); ++res_itr) {
    auto data_idx = res_itr.index();
    for (unsigned comp = 0; comp != nComps_; ++comp) {
      auto bx = res_itr.getGhostedBox();
      if constexpr (Dim == 1) {
#pragma omp parallel for default(shared) schedule(static)
        loop_box_1(bx, i) {
          at(data_idx, comp, i) = expr.at(data_idx, comp, i);
        }
      } else if constexpr (Dim == 2) {
#pragma omp parallel for default(shared) schedule(static)
        loop_box_2(bx, i, j) {
          at(data_idx, comp, i, j) = expr.at(data_idx, comp, i, j);
        }
      } else if constexpr (Dim == 3) {
#pragma omp parallel for default(shared) schedule(static)
        loop_box_3(bx, i, j, k) {
          at(data_idx, comp, i, j, k) = expr.at(data_idx, comp, i, j, k);
        }
      } else {
        assert(0);
      }
    }
  }
  UnitTimer::getInstance().end("LevelDataExpr");
  return *this;
};

template <class T, int Dim>
template <class LDExpr>
inline LevelData<T, Dim>::LevelData(const LevelDataExpr<LDExpr> &expr) :
    LevelData(expr.getMesh(),
              expr.getCentering(),
              expr.getnComps(),
              expr.getnGhost(),
              expr.getComm()){};

// if an expression does not carry mesh information,
// just return null_mesh_tag
struct null_info_tag {};

template <class T>
struct LevelDataLiteral : LevelDataExpr<LevelDataLiteral<T>> {
  T a_;
  LevelDataLiteral(const T &a) : a_(a) {}

  template <class... Ts>
  T at(int, unsigned, Ts...) const {
    return a_;
  }

  auto getMesh() const { return null_info_tag(); };

  auto getnComps() const { return null_info_tag(); }

  auto getnGhost() const { return null_info_tag(); }

  auto getCentering() const { return null_info_tag(); }

  auto getComm() const { return null_info_tag(); }
};

template <class LDExpr, class Op>
struct LevelDataUnaryOp : public LevelDataExpr<LevelDataUnaryOp<LDExpr, Op>> {
  LDExpr e1_;
  LevelDataUnaryOp(const LDExpr &e) : e1_(e) {}

  template <class... Ts>
  auto at(int data_idx, unsigned comp, Ts... args) const {
    return Op()(e1_.at(data_idx, comp, args...));
  };

  auto getMesh() const { return e1_.getMesh(); };

  auto getnComps() const { return e1_.getnComps(); }

  auto getnGhost() const { return e1_.getnGhost(); }

  auto getCentering() const { return e1_.getCentering(); }

  auto getComm() const { return e1_.getComm(); }
};

template <class LDExpr1, class LDExpr2, class Op>
struct LevelDataBinaryOp
    : public LevelDataExpr<LevelDataBinaryOp<LDExpr1, LDExpr2, Op>> {
  LDExpr1 e1_;
  LDExpr2 e2_;
  LevelDataBinaryOp(const LDExpr1 &i1, const LDExpr2 &i2) : e1_(i1), e2_(i2){};

  template <class... Ts>
  auto at(int data_idx, unsigned comp, Ts... args) const {
    return Op()(e1_.at(data_idx, comp, args...),
                e2_.at(data_idx, comp, args...));
  };

  auto getMesh() const { return getMeshByTag(e1_.getMesh()); };
  template <int Dim>
  auto &getMeshByTag(const DisjointBoxLayout<Dim> &mesh) const {
    return mesh;
  };
  auto getMeshByTag(const null_info_tag &) const { return e2_.getMesh(); };

  auto getnComps() const { return getnCompsByTag(e1_.getnComps()); };
  auto &getnCompsByTag(const unsigned &nComps) const { return nComps; };
  auto getnCompsByTag(const null_info_tag &) const { return e2_.getnComps(); };

  auto getnGhost() const { return e1_.getnGhost(); };
  auto &getnGhostByTag(const int &nGhost) const { return nGhost; };
  auto getnGhostByTag(const null_info_tag &) const { return e2_.getnGhost(); };

  auto getCentering() const { return getCenteringByTag(e1_.getCentering()); };
  auto &getCenteringByTag(const std::vector<int> &centering) const {
    return centering;
  };
  auto getCenteringByTag(const null_info_tag &) const {
    return e2_.getCentering();
  };

  auto getComm() const { return getCommByTag(e1_.getComm()); };
  auto &getCommByTag(const Communicator &comm) const { return comm; };
  auto getCommByTag(const null_info_tag &) const { return e2_.getComm(); };
};

//========================================================
// unary
template <class LDExpr>
inline auto abs(const LevelDataExpr<LDExpr> &rhs) {
  return LevelDataUnaryOp<LDExpr, NS4_Ops::absolute<>>(
      static_cast<const LDExpr &>(rhs));
};

template <class LDExpr>
inline auto operator-(const LevelDataExpr<LDExpr> &rhs) {
  return LevelDataUnaryOp<LDExpr, std::negate<>>(
      static_cast<const LDExpr &>(rhs));
};

// ========================================================
// both operands are expressions
#define BINARY_OP(OpName, Op)                                                 \
  template <class LDExpr1, class LDExpr2>                                     \
  inline auto OpName(const LevelDataExpr<LDExpr1> &lhs,                       \
                     const LevelDataExpr<LDExpr2> &rhs) {                     \
    return LevelDataBinaryOp<LDExpr1, LDExpr2, Op>(                           \
        static_cast<const LDExpr1 &>(lhs),                                    \
        static_cast<const LDExpr2 &>(rhs));                                   \
  };                                                                          \
  template <class LDExpr1, class T>                                           \
  inline auto OpName(const LevelDataExpr<LDExpr1> &lhs,                       \
                     T rhs) requires std::is_scalar<T>::value {               \
    return LevelDataBinaryOp<LDExpr1, LevelDataLiteral<T>, Op>(               \
        static_cast<const LDExpr1 &>(lhs), rhs);                              \
  };
BINARY_OP(operator+, std::plus<>);
BINARY_OP(operator-, std::minus<>);
BINARY_OP(operator*, std::multiplies<>);
#undef BINARY_OP