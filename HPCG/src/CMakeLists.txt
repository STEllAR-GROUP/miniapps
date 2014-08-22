# Copyright (c) 2013-2014 Grant Mercer
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPCG_NOHPX)
  set(SOURCES
    CG.cpp
    CG_ref.cpp
    TestCG.cpp
    ComputeResidual.cpp
    ExchangeHalo.cpp
    GenerateGeometry.cpp
    GenerateProblem.cpp
    OptimizeProblem.cpp
    ReadHpcgDat.cpp
    ReportResults.cpp
    SetupHalo.cpp
    TestSymmetry.cpp
    TestNorms.cpp
    WriteProblem.cpp
    YAML_Doc.cpp
    YAML_Element.cpp
    ComputeDotProduct.cpp
    ComputeDotProduct_ref.cpp
    mytimer.cpp
    ComputeSPMV.cpp
    ComputeSPMV_ref.cpp
    ComputeSYMGS.cpp
    ComputeSYMGS_ref.cpp
    ComputeWAXPBY.cpp
    ComputeWAXPBY_ref.cpp
    ComputeMG.cpp
    ComputeMG_ref.cpp
    ComputeProlongation.cpp
    ComputeProlongation_ref.cpp
    ComputeRestriction.cpp
    ComputeRestriction_ref.cpp
    GenerateCoarseProblem.cpp
    init.cpp
    finalize.cpp
    ../testing/main.cpp)

  include_directories(".")
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
        "/MD /Zi /D NDEBUG /Od /Ob2 /Ot /Oi /Oy /GT /GL /Gw /fp:precise /d2Zi+")
    if(MSVC13)
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
          "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zc:inline")
    endif()
  endif()
  add_hpx_executable(hpcg
    MODULE hpcg
    AUTOGLOB)
endif()


