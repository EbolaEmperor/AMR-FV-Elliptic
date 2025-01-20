#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>

// #define NDEBUG

using Real = double;

#ifndef DDIM
#define DDIM 2
#endif

#if DDIM == 2
const int SpaceDim = 2;
#elif DDIM == 3
const int SpaceDim = 3;
#endif

//=================================================

// debug output issues
#ifndef NDEBUG

extern int _dbglevel;
extern std::ostream tmpos;
extern std::ostream tmpos0;
extern std::ostream tmpos1;
#define push_dbglevel(x) _dbglevel = (_dbglevel << 4) | (x)
#define pop_dbglevel() _dbglevel = _dbglevel >> 4
#define reset_dbglevel(x) _dbglevel = (x)
#define get_dbglevel() (_dbglevel & 0x0f)

#else

#define tmpos 0 && std::cout
#define push_dbglevel(x)
#define pop_dbglevel()
#define reset_dbglevel(x)
#ifndef DBGLEVEL
#define DBGLEVEL -1
#endif
#define get_dbglevel() (DBGLEVEL)

#endif  // NDEBUG

#endif  // CONFIG_H
