#include "BRMeshRefine.h"

#include "Core/Box.h"

#include <AMRTools/AMRMeshRefine.h>
#include <Core/LinearizationHelper.h>
#include <assert.h>
#include <iomanip>
#include <utility>

// Constants for Berger-Rigoutsos algorithm
#if !defined(_BR_MIN_INFLECTION_MAG_)
#define _BR_MIN_INFLECTION_MAG_ (2)
#endif

/**
 * Note: the tags should lay inside the current AMRMesh we need to refine
 */
/**
 * @brief give the vector of boxes we need to refine according to the tags
 * with BR Method
 *
 * @param tags the tag of the grid should be refine, which is given by user
 * @param refBoxes the boxes we need to refine, which is get from the
 * BR-Algorithm
 * @param refRatio fill ratio
 */
template <int Dim>
void BRMeshRefine<Dim>::makeBoxesParallel(const LevelData<bool, Dim> &tags,
                                          Vector<Box<Dim>> &ref_boxes) const {
  //
  // Validate inputs
  //

  // The number of tagged cells in a box cannot exceed the number
  // of cells in the box so enforce an upper bound on \var{FillRatio}.
  //[NOTE: 0 or negative values are allowed -- they mean any box
  //       is acceptable.  This will probably be a mistake by the
  //       caller, but there is no obvious lower valid value for
  //       this variable (1e-6 is just as likely a mistake as 0).]
  assert(FillRatio <= 1.0);

  // clear the parameter for return
  ref_boxes.clear();

  // deal with each box seperately
  for (auto it = tags.const_begin(); it.ok(); ++it) {
    makeBoxes(it.getValidBox(), it.getData()[0], ref_boxes);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// private function
//
////////////////////////////////////////////////////////////////////////////////

template <int Dim>
void BRMeshRefine<Dim>::makeBoxes(Box<Dim> box,
                                  const Tensor<bool, Dim> &tags,
                                  Vector<Box<Dim>> &refBoxes) const {
  // get the total number of tags in the tags
  unsigned long long int NtagsInTheBox = numPtsInTheBox_(tags, box);
  //
  // Handle special cases of no tags
  //
  if (NtagsInTheBox == 0) {
    // return null box
    return;
  }
  Box<Dim> minbox = minBoxInTheBox_(tags, box);
  Box<Dim> box_hi;
  Box<Dim> &box_lo = minbox;
  // the size of the minbox should bigger than 2*2 , if the box too small, we
  // just return it
  if (NtagsInTheBox >= (minbox.volume() * FillRatio) || tooSmall_(minbox)) {
    refBoxes.push_back(minbox);
  }
  // if efficiency criterion not met
  else {
    if (splitTagsInBestDimension(tags, box_lo, box_hi)) {
      // recurse on the two halves of the Tags
      if (haveTagsInBox_(tags, box_lo)) {
        // low interval
        makeBoxes(box_lo, tags, refBoxes);
      }
      if (haveTagsInBox_(tags, box_hi)) {
        // low interval
        makeBoxes(box_hi, tags, refBoxes);
      }
    } else {
      refBoxes.push_back(minbox);
    }
  }
}

/////////////////////////splitTagsInBestDimension.2///////////////////////////////
/// @brief Compute the traces (dignatures) of the minbox in each direction,
/// and find a hole in the trace (zero value) and an inflection point (zero
/// Laplacian)in each direction; keep the best of each
/// @param a_tags_inout_lo
/// @param a_tags_hi
/// @return if the tags can be split we split the tags nd return true, else if
/// the size of the box after break is smaller than 2*2, we return false
template <int Dim>
bool BRMeshRefine<Dim>::splitTagsInBestDimension(const Tensor<bool, Dim> &tags,
                                                 //  const Box<Dim> &box,
                                                 Box<Dim> &box_lo,
                                                 Box<Dim> &box_hi) const {
  int hole_indx[Dim], best_hole_dim;  // holes in traces
  int infl_indx[Dim], best_infl_dim;  // inflaction points in traces
  int infl_val[Dim];                  // magnitudes of infl
  std::vector<int> traces[Dim];

  Box<Dim> minbox = minBoxInTheBox_(tags, box_lo);
  iVec offset = minbox.lo();
  const iVec &size = minbox.size();
  traces[0].resize(size[0], 0);
  traces[1].resize(size[1], 0);
  // D_TERM3(traces[0].resize(size[0], 0);, traces[1].resize(size[1], 0);
  //         , traces[2].resize(size[2], 0);)

  loop_box_2(minbox, i, j) {
    if (tags(i, j) == 1) {
      iVec iv = iVec{i, j} - offset;
      // D_TERM3(++traces[0][iv[0]], ; ++traces[1][iv[1]], ;
      // ++traces[2][iv[2]]);
      ++traces[0][iv[0]];
      ++traces[1][iv[1]];
    }
  }

  for (int idim = 0; idim < Dim; idim++) {
    //  The following two functions, with the a_maxBoxSize argument,
    //   help balance the tag splitting by changing where to begin
    //   searching for a split or inflection index.
    hole_indx[idim] = findSplit(traces[idim]);  // find hole
    infl_indx[idim] = findMxInflectionPoint(
        traces[idim], infl_val[idim]);  // find most infected point
  }
  // Take the highest index as the best one because we want to take as
  // large
  // a box as possible  (fewer large boxes are better than many small ones)
  best_hole_dim = maxloc(hole_indx, Dim);
  best_infl_dim = maxloc(infl_indx, Dim);

  // Split the Tag set at a hole in one of the traces, if there is one, or an
  // inflection point in the Laplacian of the traces.  Failing that, split
  // at the middle of the longest dimension of the enclosing box.
  int split_dim, split_index;
  if (hole_indx[best_hole_dim] >= 0 &&
      (abs(minbox.hi()[best_hole_dim] -
           (hole_indx[best_hole_dim] + minbox.lo()[best_hole_dim])) >= 2 &&
       abs(minbox.lo()[best_hole_dim] -
           (hole_indx[best_hole_dim] + minbox.lo()[best_hole_dim])) >= 2)) {
    split_dim = best_hole_dim;
    split_index = hole_indx[best_hole_dim] + minbox.lo()[best_hole_dim];
  } else if (infl_indx[best_infl_dim] >= 0 &&
             (abs(minbox.hi()[best_infl_dim] -
                  (infl_indx[best_infl_dim] + minbox.lo()[best_infl_dim])) >=
                  2 &&
              abs(minbox.lo()[best_hole_dim] -
                  (infl_indx[best_infl_dim] + minbox.lo()[best_infl_dim])) >=
                  2)) {
    // split at an inflection point in the trace, adjusting the trace
    // index for the offset into \var{a_tags_inout_lo} indices
    split_dim = best_infl_dim;
    split_index = infl_indx[best_infl_dim] + minbox.lo()[best_infl_dim];
  } else {
    // split on the midpoint of the longest side of \var{minbox}, rounding
    // up, allowing for \var{minbox} to have a non-zero offset
    // minbox.longside(split_dim); //[NOTE: split_dim is set by
    // \func(longside)]
    longsideRefineDirs(minbox, split_dim);
    split_index = (minbox.lo()[split_dim] + minbox.hi()[split_dim] + 1) / 2;
  }
  // check if the box after split will be too small
  if (abs(minbox.hi()[split_dim] - split_index) < 2 ||
      abs(minbox.lo()[split_dim] - split_index) < 2) {
    return false;
  }
  splitTagsInPlace(split_dim, split_index, box_lo, box_hi);
  return true;
}

/////////////////////////////////find////////////////////////////////////////////
/// @brief  Given a trace (ie. signature) of a Tag set, find a place in
/// the trace with a zero count.  Assumes the first and last elements of the
/// trace are non-zero.(promiss by minbox)
/// @param a_trace
/// @return
template <int Dim>
int BRMeshRefine<Dim>::findSplit(const std::vector<int> &a_trace) const {
  // look for a place to split at -- begin at the index after the first
  // nonzero
  for (unsigned int i = 1; i < a_trace.size() - 1; i++) {
    if (a_trace[i] == 0)
      return (i);
  }
  // nothing acceptable
  return (-1);
}

/// @brief Find the largest inflection point in the given trace
/// @param a_trace
/// @param a_maxVal
/// @return
template <int Dim>
int BRMeshRefine<Dim>::findMxInflectionPoint(const std::vector<int> &a_trace,
                                             int &a_maxVal) const {
  // first compute the discrete Laplacian of the trace
  std::vector<int> d2Trace(a_trace.size(), 0);
  for (unsigned int i = 1; i < a_trace.size() - 1; i++) {
    d2Trace[i] = a_trace[i - 1] - 2 * a_trace[i] + a_trace[i + 1];
  }
  // find inflection points and save one with the largest magnitude
  int absval, imax = 0;
  a_maxVal = -1;  // 初始化返回值
  for (unsigned int i = 2; i < a_trace.size() - 1; i++) {
    absval = abs(d2Trace[i - 1] - d2Trace[i]);
    if (d2Trace[i - 1] * d2Trace[i] < 0 &&
        absval > a_maxVal)  // 异号并且差的绝对值大于阈值
    {
      imax = i;
      a_maxVal = absval;
    }
  }

  // Find the largest inflection point, if one exists and
  // has magnitude large enough and return its index.
  // return edge-centered location of chop point.
  if (a_maxVal == -1)
    return (-1);
  else if (a_maxVal < _BR_MIN_INFLECTION_MAG_)
    return (-1);
  else
    return (imax);
}

/////////////////////////////////////split//////////////////////////////////////

template <int Dim>
void BRMeshRefine<Dim>::splitTagsInPlace(const int a_split_dir,
                                         const int a_split_indx,
                                         Box<Dim> &a_box_lo,
                                         Box<Dim> &a_box_hi) const {
  chop_(a_box_lo, a_split_dir, a_split_indx, a_box_hi);
}

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief find the maximum value in a Vector<int> or an int[] and return its
 * index.
 * @note Returns -1 if the Vector has no entries.
 */
template <int Dim>
int BRMeshRefine<Dim>::maxloc(const int *a_V, const int a_Size)
    const {  // Need only m_lowestRefineDir and m_refineDirs.
  // CH_assert( isDefined() );
  int imax = 0;
  // for ( int i=1 ; i<a_Size ; i++ ) if ( a_V[i] > a_V[imax] ) imax = i ;
  for (int i = 1; i < a_Size; i++) {
    if (a_V[i] > a_V[imax])
      imax = i;
  }
  return (imax);
}
/**
 * @brief find the longest side of a_bx and store it in a_dir
 */
template <int Dim>
int BRMeshRefine<Dim>::longsideRefineDirs(const Box<Dim> &a_bx,
                                          int &a_dir) const {
  // Need only m_lowestRefineDir and m_refineDirs.
  int maxlen = a_bx.size()[0];
  a_dir = 0;
  for (int idir = 1; idir < Dim; idir++) {
    int len = a_bx.size()[idir];
    if (len > maxlen) {
      maxlen = len;
      a_dir = idir;
    }
  }
  return maxlen;
}

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief below is some tool functions needed
 */
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief chop in place
 */
template <int Dim>
void chop_(Box<Dim> &a_box_lo,
           const int a_split_dir,
           int a_split_indx,
           Box<Dim> &a_box_hi) {
  using iVec = Vec<int, Dim>;
  iVec mv = a_box_lo.hi();
  mv[a_split_dir] = a_split_indx;
  iVec mv2 = a_box_lo.lo();
  mv2[a_split_dir] = a_split_indx + 1;

  Box<Dim> boxhi(mv2, a_box_lo.hi());
  a_box_hi = boxhi;

  Box<Dim> boxlo(a_box_lo.lo(), mv);
  a_box_lo = boxlo;
}

/**
 * @brief count the number of the tags in the boxes
 */
template <int Dim>
unsigned long long int numPtsInTheBox_(const Tensor<bool, Dim> &tags,
                                       const Box<Dim> &box) {
  unsigned long long int num = 0;

  loop_box_2(box, i, j) {
    if (tags(i, j) == 1) {
      ++num;
    }
  }
  return num;
}

/**
 * @brief find the minmum box contains a_tags in the box
 */
template <int Dim>
Box<Dim> minBoxInTheBox_(const Tensor<bool, Dim> &tags, const Box<Dim> &box) {
  auto hi = box.lo(), lo = box.hi();
  loop_box_2(box, i, j) {
    if (tags(i, j) == 1) {
      if (i < lo[0])
        lo[0] = i;
      if (i > hi[0])
        hi[0] = i;
      if (j < lo[1])
        lo[1] = j;
      if (j > hi[1])
        hi[1] = j;
    }
  }
  return Box<Dim>(lo, hi);
}

/**
 * @brief see if there are some tags in the current box region
 */
template <int Dim>
bool haveTagsInBox_(const Tensor<bool, Dim> &tags, const Box<Dim> &box) {
  loop_box_2(box, i, j) {
    if (tags(i, j) == 1) {
      return true;
    }
  }
  return false;
}

/**
 * @brief check if the size of the box is too small
 */
template <int Dim>
bool tooSmall_(const Box<Dim> &a_box) {
  for (int i = 0; i < Dim; i++) {
    if (a_box.size()[i] <= 2) {
      return true;
    }
  }
  return false;
}

template class BRMeshRefine<2>;