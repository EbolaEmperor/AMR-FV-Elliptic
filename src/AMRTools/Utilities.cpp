#include "AMRTools/Utilities.h"

constexpr char simple_path::separator;
constexpr char simple_path::the_other_sep;

//==================================================================
// Unit Timer

void UnitTimer::begin(std::string name) {
  if (!unit_.count(name)) {
    unit_[name] = time_.size();
    time_.push_back(0.);
    timer_.push_back(CPUTimer());
  }
  timer_[unit_[name]].reset();
}

void UnitTimer::end(std::string name) {
  time_[unit_[name]] += timer_[unit_[name]]();
}

void UnitTimer::reset() {
  alltimer_.reset();
  unit_.clear();
  timer_.clear();
  time_.clear();
}

void UnitTimer::report() const {
  size_t len = 8;
  for (auto it : unit_)
    len = std::max(len, it.first.size());
  mpicout << "------------------------------" << std::endl;
  mpicout << std::setw(len) << "CPU Time"
          << ": " << alltimer_() << "s." << std::endl;
  mpicout << "------------------------------" << std::endl;
  std::vector<std::pair<double, std::string>> unitTimes(time_.size());
  for (auto it : unit_)
    unitTimes[it.second] = std::make_pair(-time_[it.second], it.first);
  std::sort(unitTimes.begin(), unitTimes.end());
  for (auto it : unitTimes)
    mpicout << std::setw(len) << it.second << ": " << -it.first << "s."
            << std::endl;
}

//==================================================================
// Formatting

template <std::size_t numOfNorms>
void printConvergenceTable(
    const int *gridSize,
    const std::vector<std::array<Real, numOfNorms>> &errnorm) {
  if (ProcID() != 0)
    return;
  const int numCompHier = errnorm.size();
  const int w = 10;
  const char *ntHeader[] = {"$L^\\infty$", "$L^1$", "$L^2$"};
  mpicout << "\n" << std::setw(w) << "$h$";
  for (int n = numCompHier - 1; n >= 0; --n) {
    if (n > 0) {
      mpicout << " & " << std::setw(w) << oneOver(gridSize[n]) << " & "
              << std::setw(w) << "rate";
    } else {
      mpicout << " & " << std::setw(w) << oneOver(gridSize[n]) << " \\\\\n";
    }
  }
  std::cout.precision(2);
  for (std::size_t p = 0; p < numOfNorms; ++p) {
    // for(std::size_t p=0; p<numOfNorms/3; ++p) {
    mpicout << std::setw(w) << ntHeader[p % 3];
    for (int n = numCompHier - 1; n >= 0; --n) {
      mpicout << " & " << std::scientific << std::setw(w) << errnorm[n][p];
      if (n != 0)
        mpicout << " & " << std::fixed << std::setw(w)
                << log(errnorm[n - 1][p] / errnorm[n][p]) /
                       log(1.0 * gridSize[n] / gridSize[n - 1]);
    }
    mpicout << " \\\\" << std::endl;
  }
}

template void printConvergenceTable<1>(
    const int *,
    const std::vector<std::array<Real, 1>> &);
template void printConvergenceTable<2>(
    const int *,
    const std::vector<std::array<Real, 2>> &);
template void printConvergenceTable<3>(
    const int *,
    const std::vector<std::array<Real, 3>> &);

//==================================================================
// Input files related

std::string getTestIdentifier(const std::string &nameOfTest) {
  std::ostringstream oss;
  std::time_t t = std::time(nullptr);
  oss << nameOfTest << std::put_time(std::localtime(&t), "-%y%m%d-%H%M");
  return oss.str();
}

const lightJSON::jsonParser *getInputParser(const std::string fileName) {
  lightJSON::jsonParser *pParser = new lightJSON::jsonParser;
  std::string inputText;
  std::ifstream infile(fileName);
  assert(infile.good());
  std::stringstream ss;
  ss << infile.rdbuf();
  inputText = ss.str();
  pParser->parse(inputText.c_str());
  pParser->finish();
  return pParser;
}

//==================================================================
// Path related

void simple_copy(const simple_path &from, const simple_path &to) {
  auto _from = from;
  auto _to = to;
  using namespace std::string_literals;
#ifdef _WIN32
  std::string copyCmd = "copy "s + _from.make_preferred().c_str() + ' ' +
                        _to.make_preferred().c_str();
#else
  std::string copyCmd = "cp "s + _from.make_preferred().c_str() + ' ' +
                        _to.make_preferred().c_str();
#endif
  system(copyCmd.c_str());
}

void simple_mkdir(const simple_path &dir) {
  auto _dir = dir;
  using namespace std::string_literals;
  std::string mdCmd = "mkdir "s + _dir.make_preferred().c_str();
  system(mdCmd.c_str());
}