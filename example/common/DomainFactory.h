#ifndef DOMAINFACTORY_H
#define DOMAINFACTORY_H

#include "AMRTools/DisjointBoxLayout.h"
#include "AMRTools/ProblemDomain.h"
#include "AMRTools/Utilities.h"

#include <string>
#include <vector>

/**
 * The prototype for initializing a domain of a particular type.
 */
template <int Dim>
class DomainWrapper {
public:
  using iVec = Vec<int, Dim>;

  DomainWrapper() {}
  DomainWrapper(iVec boxSize0) : boxSize0(boxSize0) {}

  virtual ~DomainWrapper() = default;

  virtual ProblemDomain<Dim> getDomain() const = 0;

  virtual DisjointBoxLayout<Dim> getMesh() const = 0;

protected:
  iVec boxSize0;
};

//==========================================================

// forward declaration of DomainFactory<>
template <int Dim>
class DomainFactory;

/**
 * A table (a singleton) for the string-to-wrapper translation.
 */
template <int Dim>
class DomainTable {
public:
  /**
   * @return The singleton;
   */
  static DomainTable &getInstance() {
    static DomainTable singleton;
    return singleton;
  }

  /**
   * Initialize the table for all the domain types.
   */
  DomainTable();

protected:
  using Alloc = DomainWrapper<Dim> *(*)(const lightJSON::jsonNode &);
  std::map<std::string, Alloc> allocTable;
  friend class DomainFactory<Dim>;
};

//==========================================================
/**
 * The factory for generating the computational domains
 * from the input file.
 */
template <int Dim>
class DomainFactory {
public:
  using jsonNode = lightJSON::jsonNode;
  using DWR = DomainWrapper<Dim>;

  static DWR *getDomainWrapper(const jsonNode &gridNode) {
    const auto &table = DomainTable<Dim>::getInstance();
    std::string domain;
    PGET(gridNode, domain);
    auto it = table.allocTable.find(domain);
    assert(it != table.allocTable.cend());
    return (it->second)(gridNode);
  }

  static void deleteDomain(DWR *pDomain) { delete pDomain; }
};

#endif  // DOMAINFACTORY_H