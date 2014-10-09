
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
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */

#if !defined(HPCG_NOHPX)
#include <hpx/hpx_fwd.hpp>
#endif

#ifndef HPCG_NOOPENMP
#include <omp.h> // If this routine is not compiled with HPCG_NOOPENMP
#endif

#include "ComputeRestriction.hpp"
#include "ComputeRestriction_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
#if defined(HPCG_NOHPX)

int ComputeRestriction(const SparseMatrix & A, const Vector & rf) {

  return ComputeRestriction_ref(A, rf);
}

#else

#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <boost/iterator/counting_iterator.hpp>

hpx::future<void> ComputeRestriction_async(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

  typedef boost::counting_iterator<local_int_t> iterator;

  return hpx::parallel::for_each(
    hpx::parallel::par(hpx::parallel::task), iterator(0), iterator(nc),
    [rcv, rfv, Axfv, f2c](local_int_t i)
    {
      rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
    });
}

int ComputeRestriction(const SparseMatrix & A, const Vector & rf) {

  return ComputeRestriction_async(A, rf).wait(), 0;
}

#endif
