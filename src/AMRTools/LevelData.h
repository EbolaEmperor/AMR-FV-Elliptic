/**
 * @file LevelData.h
 * @author Wenchong Huang (wchung.huang@gmail.com)
 *
 * @copyright Copyright (c) 2024 Wenchong Huang
 *
 */

#pragma once

#include <AMRTools/DisjointBoxLayout.h>
#include <Core/ParallelDataBase.h>
#include <Core/Tensor.h>
#include <Core/TensorSlice.h>
#include <Core/type_traits.h>
#include <array>

template <class, class>
struct LevelDataUnaryOp;

template <class, class, class>
struct LevelDataBinaryOp;

template <class LDExpr>
class LevelDataExpr {
public:
  template <class... Ts>
  auto at(int data_idx, unsigned comp, Ts... args) const {
    return static_cast<const LDExpr &>(*this).at(data_idx, comp, args...);
  }

  auto getMesh() const {
    return static_cast<const LDExpr &>(*this).getMesh();
  };

  auto getnComps() const {
    return static_cast<const LDExpr &>(*this).getnComps();
  }
  auto getnGhost() const {
    return static_cast<const LDExpr &>(*this).getnGhost();
  }
  auto getCentering() const {
    return static_cast<const LDExpr &>(*this).getCentering();
  }
  auto getComm() const { return static_cast<const LDExpr &>(*this).getComm(); }
};

template <class T, int Dim>
class LevelData : public LevelDataExpr<const LevelData<T, Dim> &>,
                  public ParallelDataBase<FixedSizeData<T>::value,
                                          FixedSizeData<T>::value> {
public:
  template <typename U>
  using Vector = std::vector<U>;
  template <typename U, int D>
  using Array = std::array<U, D>;
  using rVec = Vec<Real, Dim>;
  using iVec = Vec<int, Dim>;
  using BaseClass =
      ParallelDataBase<FixedSizeData<T>::value, FixedSizeData<T>::value>;

private:
  /**
   * @brief The Layout where to store the data
   *
   */
  const DisjointBoxLayout<Dim> mesh_;

  /**
   * @brief The box index of each boxes that store data in present processor.
   *
   */
  Vector<int> index_;

  /**
   * @brief Number of components
   *
   */
  unsigned nComps_;

  // The data centering of each component.
  Vector<int> centering_;

  // The number of ghost layers.
  int nGhost_;

  // The datas.
  Vector<Vector<Tensor<T, Dim>>> data_;

  // The covered box(Cell-Centered) to send.
  std::map<int, std::vector<Box<Dim>>> box_to_send_;
  // The covered box(Cell-Centered) to recv.
  std::map<int, std::vector<Box<Dim>>> box_to_recv_;
  // The covered box(Cell-Centered)'s index to send.
  std::map<int, std::vector<int>> box_index_to_send_;
  // The owner's index whose covered box(Cell-Centered) is received.
  std::map<int, std::vector<int>> dest_box_index_;

public:  /// iterators
  /**
   * @brief Iterator to loop each data box.
   *
   */
  class Iterator : public BaseIterator<LevelData<T, Dim>> {
  protected:
    using BaseIterator<LevelData<T, Dim>>::object_;
    using BaseIterator<LevelData<T, Dim>>::current_;

  public:
    using BaseIterator<LevelData<T, Dim>>::BaseIterator;
    /**
     * @brief Get the Data object pointed by the iterator
     *
     * @return Vector<LevelData<T, Dim>>&
     */
    Vector<Tensor<T, Dim>> &getData() { return object_.data_[current_]; };

    /**
     * @brief Get the Box index of the current data
     */
    int getBoxID() const { return object_.index_[current_]; }

    /**
     * @brief Get the Box of the current data
     */
    Box<Dim> getBox() const {
      return object_.mesh_.getBox(object_.index_[current_]);
    };

    /**
     * @brief Get the present data's non-ghost, staggered box
     * in the same centering as centering_[comp]
     *
     * @param comp
     * @return Box<Dim>
     */
    Box<Dim> getValidBox(unsigned comp = 0) const {
      assert(comp < object_.nComps_);
      return staggerFromCellCenter(
          object_.mesh_.getBox(object_.index_[current_]),
          object_.centering_[comp]);
    };

    /**
     * @brief Get the present data's Ghosted Box centering as
     * centering_[comp]
     *
     * @param comp
     * @return Box<Dim>
     */
    Box<Dim> getGhostedBox(unsigned comp = 0) const {
      assert(comp < object_.nComps_);
      return object_.data_[current_][comp].box();
    }
  };  /// Iterator

  class ConstIterator : public BaseIterator<const LevelData<T, Dim>> {
  protected:
    using BaseIterator<const LevelData<T, Dim>>::object_;
    using BaseIterator<const LevelData<T, Dim>>::current_;

  public:
    using BaseIterator<const LevelData<T, Dim>>::BaseIterator;
    /**
     * @brief Get the Data object pointed by the iterator
     *
     * @return Vector<LevelData<T, Dim>>&
     */
    const Vector<Tensor<T, Dim>> &getData() const {
      return object_.data_[current_];
    };

    /**
     * @brief Get the Box index of the current data
     */
    int getBoxID() const { return object_.index_[current_]; }

    /**
     * @brief Get the Box of the current data
     */
    Box<Dim> getBox() const {
      return object_.mesh_.getBox(object_.index_[current_]);
    };

    /**
     * @brief Get the present data's non-ghost, staggered box
     * in the same centering as centering_[comp]
     *
     * @param comp
     * @return Box<Dim>
     */
    Box<Dim> getValidBox(unsigned comp = 0) const {
      assert(comp < object_.nComps_);
      return staggerFromCellCenter(
          object_.mesh_.getBox(object_.index_[current_]),
          object_.centering_[comp]);
    };

    /**
     * @brief Get the present data's Ghosted Box centering as
     * centering_[comp]
     *
     * @param comp
     * @return Box<Dim>
     */
    Box<Dim> getGhostedBox(unsigned comp = 0) const {
      assert(comp < object_.nComps_);
      return object_.data_[current_][comp].box();
    }
  };  /// ConstIterator

  /**
   * @brief get the beginning of the iterator
   *
   * @return Iterator
   */
  Iterator begin() { return Iterator(*this); };

  /**
   * @brief get the beginning of the start iterator
   *
   * @return ConstIterator
   */
  ConstIterator const_begin() const { return ConstIterator(*this); };

public:  /// member functions
  LevelData(const DisjointBoxLayout<Dim> &mesh,
            int centering,
            unsigned nComps = 1,
            int nGhost = 2,
            Communicator comm = MPI_COMM_WORLD) :
      LevelData(mesh, Vector<int>(nComps, centering), nComps, nGhost, comm) {}

  LevelData(const DisjointBoxLayout<Dim> &mesh,
            const Vector<int> &centering,
            unsigned nComps = 1,
            unsigned nGhost = 2,
            Communicator comm = MPI_COMM_WORLD) :
      BaseClass(comm), mesh_(mesh) {
    nComps_ = nComps;
    centering_ = centering;
    nGhost_ = nGhost;
    allocDatas();
    // initialize the MPI communication info for exchangeAll().
    setBoxToSendAndRecv();
    BaseClass::initialize();
  }

  LevelData(const LevelData<T, Dim> &other) :
      BaseClass(other),
      mesh_(other.mesh_),
      index_(other.index_),
      nComps_(other.nComps_),
      centering_(other.centering_),
      nGhost_(other.nGhost_),
      data_(other.data_),
      box_to_send_(other.box_to_send_),
      box_to_recv_(other.box_to_recv_),
      box_index_to_send_(other.box_index_to_send_),
      dest_box_index_(other.dest_box_index_) {}

  LevelData<T, Dim> &operator=(const T a);
  template <class LDExpr>
  LevelData(const LevelDataExpr<LDExpr> &expr);
  // =============================================== copy assignment
  LevelData<T, Dim> &operator=(const LevelData<T, Dim> &rhs);

  template <class LDExpr>
  LevelData<T, Dim> &operator=(const LevelDataExpr<LDExpr> &expr);

  //================================================ move ctor and move
  // assignment
  // LevelData(LevelData<T, Dim> &&rhs) noexcept;

  // LevelData<T, Dim> &operator=(LevelData<T, Dim> &&rhs) noexcept;
  /**
   * @brief Get the Mesh object
   *
   * @return const DisjointBoxLayout<Dim>&
   */
  const DisjointBoxLayout<Dim> &getMesh() const { return mesh_; };

  /**
   * @brief get the number of data boxes.
   *
   * @return unsigned
   */
  unsigned size() const { return data_.size(); };

  /**
   * @brief get the number of components.
   *
   * @return unsigned
   */
  unsigned getnComps() const { return nComps_; };

  /**
   * @brief get the number of ghost layers.
   *
   * @return int
   */
  int getnGhost() const { return nGhost_; };

  /**
   * @brief Get the Centering object
   *
   * @return std::vector<int>
   */
  std::vector<int> getCentering() const { return centering_; }

  /**
   * @brief Get the Comm object
   *
   * @return Communicator
   */
  Communicator getComm() const { return BaseClass::comm_; }
  /**
   * @brief Get the Centering object of the comp's component
   *
   * @param comp which component
   * @return int the centering
   */
  int getCentering(unsigned comp) const {
    assert(comp < nComps_);
    return centering_[comp];
  };

  /**
   * @brief Get the data
   *
   * @param box_id
   * @return Vector<Tensor<T, Dim>>&
   */
  Vector<Tensor<T, Dim>> &getBoxData(int box_id) {
    auto it = std::find(index_.begin(), index_.end(), box_id);
    assert(it != index_.end());
    unsigned index = std::distance(index_.begin(), it);
    return data_.at(index);
  };

  /**
   * @brief Get the Box Data object
   *
   * @param box_id
   * @return const Vector<Tensor<T, Dim>>&
   */
  const Vector<Tensor<T, Dim>> &getBoxData(int box_id) const {
    auto it = std::find(index_.begin(), index_.end(), box_id);
    assert(it != index_.end());
    unsigned index = std::distance(index_.begin(), it);
    return data_.at(index);
  };

  // Exchange the data with the LevelData in other process.
  // Using the exchanged datas to fill corresponding ghost cells.
  void exchangeAll();

  /**
   * @brief memset all datas with c
   * @note we use standard memset inside, which is bytewise
   */
  void memset(int c);

private:
  // Alloc memory for datas_
  void allocDatas();

  void setBoxToSendAndRecv();

  /**
   * @brief Set the Data Length Info object to sync
   *
   */
  void setDataLengthInfo() override;

  /**
   * @brief Linearize the data to send.
   *
   */
  void linearIn() override;

  /**
   * @brief Rebuild the data received.
   *
   */
  void linearOut() override;

protected:
  template <class TExpr>
  friend class LevelDataExpr;
  template <class, class>
  friend struct LevelDataUnaryOp;
  template <class, class, class>
  friend struct LevelDataBinaryOp;
  // just for LevelDataExpr
  const T &at(int data_idx, unsigned comp, int i) const {
    return data_[data_idx][comp](i);
  }
  T &at(int data_idx, unsigned comp, int i) {
    return data_[data_idx][comp](i);
  }
  const T &at(int data_idx, unsigned comp, int i, int j) const {
    return data_[data_idx][comp](i, j);
  }
  T &at(int data_idx, unsigned comp, int i, int j) {
    return data_[data_idx][comp](i, j);
  }
  const T &at(int data_idx, unsigned comp, int i, int j, int k) const {
    return data_[data_idx][comp](i, j, k);
  }
  T &at(int data_idx, unsigned comp, int i, int j, int k) {
    return data_[data_idx][comp](i, j, k);
  }
};

template <typename T, int Dim>
std::ostream &operator<<(std::ostream &os, const LevelData<T, Dim> &rhs) {
  for (auto it = rhs.const_begin(); it.ok(); ++it)
    for (unsigned i = 0; i < rhs.getnComps(); ++i)
      os << it.getData()[i] << std::endl;
  return os;
}