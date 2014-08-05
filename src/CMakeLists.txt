# Copyright (c) 2013-2014 Grant Mercer
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPCG_HPX)
  set(SOURCES
    src/CG.cpp
    src/CG_ref.cpp
    src/TestCG.cpp
    src/ComputeResidual.cpp
    src/ExchangeHalo.cpp
    src/GenerateGeometry.cpp
    src/GenerateProblem.cpp
    src/OptimizeProblem.cpp
    src/ReadHpcgDat.cpp
    src/ReportResults.cpp
    src/SetupHalo.cpp
    src/TestSymmetry.cpp
    src/TestNorms.cpp
    src/WriteProblem.cpp
    src/YAML_Doc.cpp
    src/YAML_Element.cpp
    src/ComputeDotProduct.cpp
    src/ComputeDotProduct_ref.cpp
    src/mytimer.cpp
    src/ComputeSPMV.cpp
    src/ComputeSPMV_ref.cpp
    src/ComputeSYMGS.cpp
    src/ComputeSYMGS_ref.cpp
    src/ComputeWAXPBY.cpp
    src/ComputeWAXPBY_ref.cpp
    src/ComputeMG.cpp
    src/ComputeMG_ref.cpp
    src/ComputeProlongation.cpp
    src/ComputeProlongation_ref.cpp
    src/ComputeRestriction.cpp
    src/ComputeRestriction_ref.cpp
    src/GenerateCoarseProblem.cpp
    src/init.cpp
    src/finalize.cpp
    testing/main.cpp)

  add_executable(hpcg ${SOURCES})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  if(HPCG_MPI)
     target_link_libraries(hpcg ${MPI_LIBRARIES})
     if(MPI_COMPILE_FLAGS)
         set_target_properties(hpcg PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
     endif()
     if(MPI_LINK_FLAGS)
         set_target_properties(hpcg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
     endif()
  endif()
else()
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd4244")
    # we can't use -O2 with MSVC
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "/MD /Zi /D NDEBUG /Od /Ob2 /Ot /Oi /Oy /GT /GL /fp:precise")
  endif()
  add_hpx_executable(hpcg
    MODULE hpcg
    AUTOGLOB
    COMPONENT_DEPENDENCIES iostreams)
endif()


