#ifndef BOX_H
#define BOX_H

#include "Vec.h"

#include <Core/LinearizationHelper.h>

enum Centering {
  CellCenter = -1,
  NodeCenter = -2,
  FaceCenter0 = 0,
  FaceCenter1 = 1
};

enum FaceOrdering { LeftFace = 0, RightFace = 1, DownFace = 2, UpFace = 3 };

template <int Dim>
class Box {
public:
  using iVec = Vec<int, Dim>;

  Box() : corners{0, -1} {};

  Box(const iVec &_lo, const iVec &_hi) : corners{_lo, _hi} {}

  const iVec &lo() const { return corners[0]; }
  const iVec &hi() const { return corners[1]; }

  iVec size() const { return hi() - lo() + 1; }

  // deformation
public:
  Box<Dim> operator+(const iVec &delta) const {
    return Box<Dim>(lo() + delta, hi() + delta);
  }
  Box<Dim> operator-(const iVec &delta) const {
    return Box<Dim>(lo() - delta, hi() - delta);
  }

  bool operator==(const Box<Dim> &rhs) const {
    return lo() == rhs.lo() && hi() == rhs.hi();
  }

  bool operator!=(const Box<Dim> &rhs) const { return !(*this == rhs); }

  // positive delta for inflation, negative delta for shrinkage
  Box<Dim> inflate(const iVec &delta) const {
    return Box<Dim>(lo() - delta, hi() + delta);
  }

  Box<Dim> operator&(const Box &rhs) const {
    auto l = max(lo(), rhs.lo());
    auto h = min(hi(), rhs.hi());
    return Box<Dim>(l, h);
  }

  //  Box<Dim> refine() const { return Box<Dim>(corners[0]*2,
  //  corners[1]*2+1); } Box<Dim> coarsen() const { return
  //  Box<Dim>(corners[0]/2, corners[1]/2); }

public:
  bool empty() const { return min_of(hi() - lo()) < 0; }

  bool contain(const iVec &pos) const {
    return (min_of(hi() - pos) >= 0) && (min_of(pos - lo()) >= 0);
  }

  bool contain(const Box<Dim> &rhs) const {
    return contain(rhs.lo()) && contain(rhs.hi());
  }

  int volume() const { return prod(size()); }

  Box<Dim> lowSideBox(unsigned centering) const {
    iVec high = hi();
    high[centering] = lo()[centering];
    return Box<Dim>(lo(), high);
  }

  Box<Dim> highSideBox(unsigned centering) const {
    iVec low = lo();
    low[centering] = hi()[centering];
    return Box<Dim>(low, hi());
  }

  Box<Dim> grow(int nGhost) const {
    return Box<Dim>(lo() - nGhost, hi() + nGhost);
  }

  Box<Dim> grow(int nGhost, int d) const {
    iVec low = lo(), high = hi();
    low[d] -= nGhost;
    high[d] += nGhost;
    return Box<Dim>(low, high);
  }

  Box<Dim> getRefined(int ratio, int cent = CellCenter) const {
    if (cent == CellCenter)
      return Box<Dim>(lo() * ratio, hi() * ratio + (ratio - 1));
    else if (cent == NodeCenter)
      return Box<Dim>(lo() * ratio, hi() * ratio);
    else {  // Face-Center
      auto high = hi();
      high[cent] *= ratio;
      for (int i = 0; i < Dim; i++)
        if (i != cent)
          high[i] = high[i] * ratio + (ratio - 1);
      return Box<Dim>(lo() * ratio, high);
    }
  }

  Box<Dim> getCoarsened(int ratio, int cent = CellCenter) const {
    if (cent == CellCenter)
      return Box<Dim>(lo() / ratio, (hi() + 1) / ratio - 1);
    else if (cent == NodeCenter)
      return Box<Dim>(lo() / ratio, hi() / ratio);
    else {  // Face-Center
      auto high = hi();
      high[cent] /= ratio;
      for (int i = 0; i < Dim; i++)
        if (i != cent)
          high[i] = (high[i] + 1) / ratio - 1;
      return Box<Dim>(lo() / ratio, high);
    }
  }

  std::vector<Box<Dim>> divideInto(int n, int dir = 0) const {
    std::vector<Box<Dim>> divBoxes(n);
    int len = size()[dir];
    for (int i = 0; i < n; ++i) {
      iVec low = lo(), high = hi();
      high[dir] = low[dir] + (i + 1) * len / n - 1;
      low[dir] += i * len / n;
      divBoxes[i] = Box<Dim>(low, high);
    }
    return divBoxes;
  }

  friend std::vector<Box<Dim>>
  getSplittedBoxes(const std::vector<Box<Dim>> &boxes, int n, int dir = 0) {
    std::vector<Box<Dim>> splitted;
    for (auto &box : boxes) {
      auto tmpboxes = box.divideInto(n, dir);
      splitted.insert(splitted.end(), tmpboxes.begin(), tmpboxes.end());
    }
    return splitted;
  }

  Box<Dim> shift(const iVec &mv) const {
    return Box<Dim>(lo() + mv, hi() + mv);
  }

  friend std::ostream &operator<<(std::ostream &os, const Box &bx) {
    os << bx.lo() << " " << bx.hi();
    return os;
  }

  static inline void linearIntoNewBuf(const Box<Dim> &input,
                                      std::vector<char> *buf);

  static inline void linearIntoOldBuf(const Box<Dim> &input,
                                      std::vector<char> *buf,
                                      size_t *pos);

  static inline void linearOut(const std::vector<char> &buf,
                               size_t *pos,
                               Box<Dim> *res);

protected:
  iVec corners[2];
};

template <int Dim>
inline Box<1> getComp(const Box<Dim> &bx, int comp) {
  return Box<1>(bx.lo()[comp], bx.hi()[comp]);
}

//==================================================

template <int Dim>
inline Box<Dim + 1> enlarge(const Box<Dim> &bx, const Box<1> &r, int k = Dim) {
  return Box<Dim + 1>(enlarge(bx.lo(), r.lo()[0], k),
                      enlarge(bx.hi(), r.hi()[0], k));
}

template <int Dim>
inline Box<Dim - 1> reduce(const Box<Dim> &bx, int D = Dim - 1) {
  return Box<Dim - 1>(reduce(bx.lo(), D), reduce(bx.hi(), D));
}

//==================================================

#define loop_box_1(bx, i0) for (int i0 = bx.lo()[0]; i0 <= bx.hi()[0]; ++i0)

#define loop_sz_1(sz, i0) for (int i0 = 0; i0 < sz[0]; ++i0)

#define loop_box_2(bx, i0, i1)                                                \
  for (int i1 = bx.lo()[1]; i1 <= bx.hi()[1]; ++i1)                           \
    for (int i0 = bx.lo()[0]; i0 <= bx.hi()[0]; ++i0)

#define loop_sz_2(sz, i0, i1)                                                 \
  for (int i1 = 0; i1 < sz[1]; ++i1)                                          \
    for (int i0 = 0; i0 < sz[0]; ++i0)

#define loop_box_3(bx, i0, i1, i2)                                            \
  for (int i2 = bx.lo()[2]; i2 <= bx.hi()[2]; ++i2)                           \
    for (int i1 = bx.lo()[1]; i1 <= bx.hi()[1]; ++i1)                         \
      for (int i0 = bx.lo()[0]; i0 <= bx.hi()[0]; ++i0)

#define loop_sz_3(sz, i0, i1, i2)                                             \
  for (int i2 = 0; i2 < sz[2]; ++i2)                                          \
    for (int i1 = 0; i1 < sz[1]; ++i1)                                        \
      for (int i0 = 0; i0 < sz[0]; ++i0)

#define loop_box_4(bx, i0, i1, i2, i3)                                        \
  for (int i3 = bx.lo()[3]; i3 <= bx.hi()[3]; ++i3)                           \
    for (int i2 = bx.lo()[2]; i2 <= bx.hi()[2]; ++i2)                         \
      for (int i1 = bx.lo()[1]; i1 <= bx.hi()[1]; ++i1)                       \
        for (int i0 = bx.lo()[0]; i0 <= bx.hi()[0]; ++i0)

#define loop_sz_4(sz, i0, i1, i2, i3)                                         \
  for (int i3 = 0; i3 < sz[3]; ++i3)                                          \
    for (int i2 = 0; i2 < sz[2]; ++i2)                                        \
      for (int i1 = 0; i1 < sz[1]; ++i1)                                      \
        for (int i0 = 0; i0 < sz[0]; ++i0)

template <int Dim>
inline Box<Dim> staggerFromCellCenter(const Box<Dim> &bx, int centering_) {
  auto lo = bx.lo(), hi = bx.hi();
  if (centering_ == NodeCenter)
    hi = hi + 1;
  else if (centering_ != CellCenter) {
    ++hi[centering_];
  }
  return Box<Dim>(lo, hi);
}

template <int Dim>
void Box<Dim>::linearIntoNewBuf(const Box<Dim> &input,
                                std::vector<char> *buf) {
  LinearizationHelper::linearIntoNewBuf(input.lo(), buf);
  LinearizationHelper::linearIntoNewBuf(input.hi(), buf);
};

template <int Dim>
void Box<Dim>::linearIntoOldBuf(const Box<Dim> &input,
                                std::vector<char> *buf,
                                size_t *pos) {
  LinearizationHelper::linearIntoOldBuf(input.lo(), buf, pos);
  LinearizationHelper::linearIntoOldBuf(input.hi(), buf, pos);
};

template <int Dim>
void Box<Dim>::linearOut(const std::vector<char> &buf,
                         size_t *pos,
                         Box<Dim> *res) {
  Vec<int, Dim> lo, hi;
  LinearizationHelper::linearOut(buf, pos, &lo);
  LinearizationHelper::linearOut(buf, pos, &hi);
  *res = Box<Dim>(lo, hi);
};
#endif  // BOX_H
