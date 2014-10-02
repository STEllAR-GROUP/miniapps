
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#if !defined(HPCG_NOHPX)
#include <hpx/hpx_fwd.hpp>
#endif

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
#if defined(HPCG_NOHPX)

int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  isOptimized = false;
  return(ComputeDotProduct_ref(n, x, y, result, time_allreduce));
}

#else

#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_transform_reduce.hpp>

#include <boost/iterator/counting_iterator.hpp>

hpx::future<double> ComputeDotProduct_async(
    const local_int_t n, const Vector & x, const Vector & y,
    double & time_allreduce) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double * xv = x.values;
  double * yv = y.values;

  typedef boost::counting_iterator<local_int_t> iterator;

  if (yv == xv) {
    return
      hpx::parallel::transform_reduce(
          hpx::parallel::par(hpx::parallel::task), iterator(0), iterator(n), 0.0,
          std::plus<double>(),
          [xv](local_int_t i)
          {
              return xv[i] * xv[i];
          });
  }

  return
    hpx::parallel::transform_reduce(
      hpx::parallel::par(hpx::parallel::task), iterator(0), iterator(n), 0.0,
      std::plus<double>(),
      [xv, yv](local_int_t i)
      {
          return xv[i] * yv[i];
      });
}

int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  isOptimized = true;
  result = ComputeDotProduct_async(n, x, y, time_allreduce).get();
  return 0;
}

#endif
