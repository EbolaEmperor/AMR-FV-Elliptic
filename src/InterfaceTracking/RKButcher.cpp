#include "RKButcher.h"

constexpr int ButcherTableau<RK::ERK, RK::ForwardEuler>::order;
constexpr int ButcherTableau<RK::ERK, RK::ForwardEuler>::nStages;
constexpr Real ButcherTableau<RK::ERK, RK::ForwardEuler>::a
    [][ButcherTableau<RK::ERK, RK::ForwardEuler>::nStages];
constexpr Real ButcherTableau<RK::ERK, RK::ForwardEuler>::b[];
constexpr Real ButcherTableau<RK::ERK, RK::ForwardEuler>::c[];

constexpr int ButcherTableau<RK::ERK, RK::ClassicRK4>::order;
constexpr int ButcherTableau<RK::ERK, RK::ClassicRK4>::nStages;
constexpr Real ButcherTableau<RK::ERK, RK::ClassicRK4>::a
    [][ButcherTableau<RK::ERK, RK::ClassicRK4>::nStages];
constexpr Real ButcherTableau<RK::ERK, RK::ClassicRK4>::b[];
constexpr Real ButcherTableau<RK::ERK, RK::ClassicRK4>::c[];

constexpr int ButcherTableau<RK::DIRK, RK::SDIRK2>::order;
constexpr int ButcherTableau<RK::DIRK, RK::SDIRK2>::nStages;
constexpr Real ButcherTableau<RK::DIRK, RK::SDIRK2>::a
    [][ButcherTableau<RK::DIRK, RK::SDIRK2>::nStages];
constexpr Real ButcherTableau<RK::DIRK, RK::SDIRK2>::b[];
constexpr Real ButcherTableau<RK::DIRK, RK::SDIRK2>::c[];

constexpr int ButcherTableau<RK::DIRK, RK::ESDIRK4>::order;
constexpr int ButcherTableau<RK::DIRK, RK::ESDIRK4>::nStages;
constexpr Real ButcherTableau<RK::DIRK, RK::ESDIRK4>::a
    [][ButcherTableau<RK::DIRK, RK::ESDIRK4>::nStages];
constexpr Real ButcherTableau<RK::DIRK, RK::ESDIRK4>::b[];
constexpr Real ButcherTableau<RK::DIRK, RK::ESDIRK4>::c[];
