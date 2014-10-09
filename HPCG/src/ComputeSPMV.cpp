
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#if !defined(HPCG_NOHPX)
#include <hpx/hpx_fwd.hpp>
#endif

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
#if defined(HPCG_NOHPX)

int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = false;
  return(ComputeSPMV_ref(A, x, y));
}

#else

#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <boost/iterator/counting_iterator.hpp>

hpx::future<void> ComputeSPMV_async( const SparseMatrix & A, /*const*/ Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NOMPI
    ExchangeHalo(A,x);
#endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

  typedef boost::counting_iterator<local_int_t> iterator;

  return hpx::parallel::for_each(
    hpx::parallel::par(hpx::parallel::task), iterator(0), iterator(nrow),
    [xv, yv, &A](local_int_t i) {
      double sum = 0.0;
      const double * const cur_vals = A.matrixValues[i];
      const local_int_t * const cur_inds = A.mtxIndL[i];
      const int cur_nnz = A.nonzerosInRow[i];

      for (int j=0; j< cur_nnz; j++)
        sum += cur_vals[j]*xv[cur_inds[j]];
      yv[i] = sum;
    });
}

int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

  A.isSpmvOptimized = true;
  return ComputeSPMV_async(A, x, y).wait(), 0;
}

#endif
