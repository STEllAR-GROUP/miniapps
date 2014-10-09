
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#if !defined(HPCG_NOHPX)
#include <hpx/hpx_fwd.hpp>
#endif

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
#if defined(HPCG_NOHPX)

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = false;
  return(ComputeWAXPBY_ref(n, alpha, x, beta, y, w));
}

#else

#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <boost/iterator/counting_iterator.hpp>

hpx::future<void> ComputeWAXPBY_async(
    const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;

  typedef boost::counting_iterator<local_int_t> iterator;

  if (alpha==1.0) {
    return hpx::parallel::for_each(
     hpx::parallel::par(hpx::parallel::task), iterator(0), iterator(n),
      [xv, yv, beta, wv](local_int_t i)
      {
        wv[i] = xv[i] + beta * yv[i];
      });
  }

  if (beta==1.0) {
    return hpx::parallel::for_each(
     hpx::parallel::par(hpx::parallel::task), iterator(0), iterator(n),
      [xv, yv, alpha, wv](local_int_t i)
      {
        wv[i] = alpha * xv[i] + yv[i];
      });
  }

  return hpx::parallel::for_each(
   hpx::parallel::par(hpx::parallel::task), iterator(0), iterator(n),
    [xv, yv, alpha, beta, wv](local_int_t i)
    {
      wv[i] = alpha * xv[i] + beta * yv[i];
    });
}

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  isOptimized = true;
  return ComputeWAXPBY_async(n, alpha, x, beta, y, w).wait(), 0;
}

#endif
