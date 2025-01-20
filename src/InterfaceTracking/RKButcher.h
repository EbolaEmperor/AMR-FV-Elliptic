#ifndef _RKBUTCHER_H_
#define _RKBUTCHER_H_

#include "Core/Config.h"

#include <cmath>

namespace RK {
enum Type_Major { ERK = 0, DIRK, ARK, nRK_Family };

enum Type_Minor { ForwardEuler = 0, ClassicRK4, SDIRK2, ESDIRK4, nRK_Type };
}  // namespace RK

template <RK::Type_Major Type1, RK::Type_Minor Type2>
struct ButcherTableau;

template <>
struct ButcherTableau<RK::ERK, RK::ForwardEuler> {
  static constexpr int order = 1;
  static constexpr int nStages = 1;
  static constexpr Real a[][1] = {{0}};
  static constexpr Real b[] = {1};
  static constexpr Real c[] = {0};
};

template <>
struct ButcherTableau<RK::ERK, RK::ClassicRK4> {
  static constexpr int order = 4;
  static constexpr int nStages = 4;
  static constexpr Real a[][4] = {{0, 0, 0, 0},
                                  {0.5, 0, 0, 0},
                                  {0, 0.5, 0, 0},
                                  {0, 0, 1, 0}};
  static constexpr Real b[] = {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6};
  static constexpr Real c[] = {0.0, 0.5, 0.5, 1.0};
};

static constexpr Real sqrt2 = 1.41421356237309514547462185873883e+00;

template <>
struct ButcherTableau<RK::DIRK, RK::SDIRK2> {
  static constexpr int order = 2;
  static constexpr int nStages = 2;
  static constexpr Real a[][2] = {{1 - sqrt2 / 2, 0},
                                  {sqrt2 / 2.0, 1 - sqrt2 / 2}};
  static constexpr Real b[] = {sqrt2 / 2, 1 - sqrt2 / 2};
  static constexpr Real c[] = {1 - sqrt2 / 2, 1};
};

template <>
struct ButcherTableau<RK::DIRK, RK::ESDIRK4> {
  static constexpr int order = 4;
  static constexpr int nStages = 6;
  static constexpr Real a[][6] = {
      {0, 0, 0, 0, 0, 0},
      {1.0 / 4, 1.0 / 4, 0, 0, 0, 0},
      {(1 - sqrt2) / 8, (1 - sqrt2) / 8, 1.0 / 4, 0, 0, 0},
      {(5 - 7 * sqrt2) / 64,
       (5 - 7 * sqrt2) / 64,
       7 * (1 + sqrt2) / 32,
       1.0 / 4,
       0,
       0},
      {(-13796 - 54539 * sqrt2) / 125000,
       (-13796 - 54539 * sqrt2) / 125000,
       (506605 + 132109 * sqrt2) / 437500,
       166 * (-97 + 376 * sqrt2) / 109375,
       1.0 / 4,
       0},
      {(1181 - 987 * sqrt2) / 13782,
       (1181 - 987 * sqrt2) / 13782,
       47 * (-267 + 1783 * sqrt2) / 273343,
       -16 * (-22922 + 3525 * sqrt2) / 571953,
       -15625 * (97 + 376 * sqrt2) / 90749876,
       1.0 / 4}};
  static constexpr Real b[] = {(1181 - 987 * sqrt2) / 13782,
                               (1181 - 987 * sqrt2) / 13782,
                               47 * (-267 + 1783 * sqrt2) / 273343,
                               -16 * (-22922 + 3525 * sqrt2) / 571953,
                               -15625 * (97 + 376 * sqrt2) / 90749876,
                               1.0 / 4};
  static constexpr Real c[] =
      {0, 1.0 / 2, (2 - sqrt2) / 4, 5.0 / 8, 26.0 / 25, 1};
};

#endif