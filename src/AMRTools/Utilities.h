#ifndef TESTUTILITY_H
#define TESTUTILITY_H

#include "Core/MPI.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#ifdef NDEBUG
#define LIGHTJSON_RT_TYPE_CHECK 0
#define LIGHTJSON_RT_BOUND_CHECK 0
#else
#define LIGHTJSON_RT_TYPE_CHECK 1
#define LIGHTJSON_RT_BOUND_CHECK 1
#endif
#define LIGHTJSON_INTEROPERABILITY 1
#include "lightJSON/lightJSON.h"

//==================================================================
// Timer

struct CPUTimer {
  using HRC = std::chrono::high_resolution_clock;
  HRC::time_point start;
  CPUTimer() { reset(); }
  void reset() { start = HRC::now(); }
  double operator()() const {
    HRC::duration e = HRC::now() - start;
    return 1e-9 *
           std::chrono::duration_cast<std::chrono::nanoseconds>(e).count();
  }
};

class UnitTimer {
private:
  CPUTimer alltimer_;
  std::map<std::string, int> unit_;
  std::vector<CPUTimer> timer_;
  std::vector<double> time_;

public:
  static UnitTimer &getInstance() {
    static UnitTimer singleton;
    return singleton;
  }

public:
  void begin(std::string name);
  void end(std::string name);
  void reset();
  void report() const;

private:
  UnitTimer() = default;
  UnitTimer(const UnitTimer &) = default;
  UnitTimer &operator=(const UnitTimer &) = default;
  ~UnitTimer() = default;
};

//==================================================================
// Formatting

template <class T>
inline std::string oneOver(const T &denom) {
  std::ostringstream oss;
  oss << "1/" << denom;
  return oss.str();
}

template <std::size_t numOfNorms>
void printConvergenceTable(
    const int *gridSize,
    const std::vector<std::array<Real, numOfNorms>> &errnorm);

//==================================================================
// Input files related

std::string getTestIdentifier(const std::string &nameOfTest);

const lightJSON::jsonParser *getInputParser(const std::string fileName);

#define PGET(_node_, _var_) _node_[#_var_].get(_var_)

//==================================================================
// Path related

class simple_path {
public:
#ifdef _WIN32
  static constexpr char separator = '\\';
  static constexpr char the_other_sep = '/';
#else
  static constexpr char separator = '/';
  static constexpr char the_other_sep = '\\';
#endif

  /// Initialize an empty path.
  simple_path() = default;

  /// Initialize a given path.
  simple_path(const std::string &_initPath) : _path(_initPath) {}

  /// Append an element
  simple_path &append(const std::string &element) {
    if (!_path.empty() && _path.back() != separator)
      _path.append(1, separator);
    _path.append(element);
    return *this;
  }

  /// Go to the parent directory.
  simple_path parent_path() const {
    auto sep_pos = _path.find_last_of(separator);
    if (sep_pos != std::string::npos) {
      return std::string(_path, 0, std::string::npos);
    } else {
      return simple_path();
    }
  }

  /// Enforce the preferred separator.
  simple_path &make_preferred() {
    std::replace(_path.begin(), _path.end(), the_other_sep, separator);
    return *this;
  }

  /// Conversion to plain string.
  const char *c_str() const { return _path.c_str(); }

  unsigned size() const { return _path.size(); }

protected:
  std::string _path;
};

void simple_copy(const simple_path &from, const simple_path &to);

void simple_mkdir(const simple_path &dir);

#endif  // TESTUTILITY_H