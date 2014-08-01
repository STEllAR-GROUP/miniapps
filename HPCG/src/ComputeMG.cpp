
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
 @file ComputeMG.cpp

 HPCG routine
 */

#if !defined(HPCG_NOHPX)
#include <hpx/hpx_fwd.hpp>
#endif

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
#if defined(HPCG_NOHPX)

int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  A.isMgOptimized = false;
  return(ComputeMG_ref(A, r, x));

}

#else

#include <hpx/include/lcos.hpp>

#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction.hpp"
#include "ComputeProlongation.hpp"

hpx::future<int> ComputeMG_async(const SparseMatrix  & A, const Vector & r, Vector & x) {

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0) hpx::make_ready_future(ierr);

    ierr = ComputeSPMV(A, x, *A.mgData->Axf);
    if (ierr!=0) hpx::make_ready_future(ierr);

    // Perform restriction operation using simple injection
    ierr = ComputeRestriction(A, r);
    if (ierr!=0) hpx::make_ready_future(ierr);

    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc);
    if (ierr!=0) hpx::make_ready_future(ierr);

    ierr = ComputeProlongation(A, x);
    if (ierr!=0) hpx::make_ready_future(ierr);

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);
  }
  else {
    ierr = ComputeSYMGS(A, r, x);
  }

  return hpx::make_ready_future(ierr);
}

int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  A.isMgOptimized = true;
  return ComputeMG_async(A, r, x).get();

}

#endif
