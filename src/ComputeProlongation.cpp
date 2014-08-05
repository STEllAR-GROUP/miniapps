
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
 @file ComputeProlongation.cpp

 HPCG routine
 */

#if !defined(HPCG_NOHPX)
#include <hpx/hpx_fwd.hpp>
#endif

#ifndef HPCG_NOOPENMP
#include <omp.h> // If this routine is not compiled with HPCG_NOOPENMP
#endif

#include "ComputeProlongation.hpp"
#include "ComputeProlongation_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
#if defined(HPCG_NOHPX)

int ComputeProlongation(const SparseMatrix & Af, Vector & xf) {

  return ComputeProlongation_ref(Af, xf);
}

#else

#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <boost/iterator/counting_iterator.hpp>

hpx::future<void> ComputeProlongation_async(const SparseMatrix & Af, Vector & xf) {

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  typedef boost::counting_iterator<local_int_t> iterator;

  // This loop is safe to vectorize
  return hpx::parallel::for_each(
    hpx::parallel::task, iterator(0), iterator(nc),
    [xfv, xcv, f2c](local_int_t i)
    {
      xfv[f2c[i]] += xcv[i];
    });
}

int ComputeProlongation(const SparseMatrix & Af, Vector & xf) {

  return ComputeProlongation_async(Af, xf).wait(), 0;
}

#endif
