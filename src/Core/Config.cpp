#include "Config.h"

#include <fstream>

#ifndef NDEBUG
int _dbglevel = 0;
std::ofstream _tmpos("tmpos.dat", std::ios::binary);
std::ostream tmpos(_tmpos.rdbuf());
std::ofstream _tmpos0("tmpos0.dat", std::ios::binary);
std::ostream tmpos0(_tmpos0.rdbuf());
std::ofstream _tmpos1("tmpos1.dat", std::ios::binary);
std::ostream tmpos1(_tmpos1.rdbuf());
#endif  // NDEBUG
