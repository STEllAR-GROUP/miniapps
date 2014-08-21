# Copyright (c) 2013-2014 Grant Mercer
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Find and include HPX settings
if(NOT HPX_ROOT)
  message(FATAL "HPX_ROOT not set, unable to find HPX!")
endif()

# Require a minimum version of CMake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

#This project is C++ based.
project(hpcg CXX)

# This adds the HPX cmake configuration directory to the search path.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)

if(DEFINED HPCG_NOHPX)
  set(HPCG_NOHPX ${HPCG_NOHPX} CACHE BOOL "Don't use HPX" FORCE)
else()
  set(HPCG_NOHPX OFF CACHE BOOL "Don't use HPX" FORCE)
endif()

if(DEFINED HPCG_NOOPENMP)
  set(HPCG_NOOPENMP ${HPCG_NOOPENMP} CACHE BOOL "Don't use OpenMP" FORCE)
else()
  set(HPCG_NOOPENMP ON CACHE BOOL "Don't use OpenMP" FORCE)
endif()

if(DEFINED HPCG_NOMPI)
  set(HPCG_NOMPI ${HPCG_NOMPI} CACHE BOOL "Don't use MPI" FORCE)
else()
  set(HPCG_NOMPI ON CACHE BOOL "Don't use MPI" FORCE)
endif()

set(HPCG_DEBUG OFF CACHE BOOL "Enable additional debug output" FORCE)

# Instruct cmake to find the HPX settings
if(NOT HPCG_NOHPX)
  find_package(HPX REQUIRED)
  if(NOT MSVC)
    set(HPX_RPATH "${CMAKE_BINARY_DIR}/lib/hpx:${HPX_RPATH}:${CMAKE_INSTALL_PREFIX}/lib/hpx")
  endif()
  include_directories(${HPX_INCLUDE_DIR})
  link_directories(${HPX_LIBRARY_DIR})
else()
   add_definitions("-DHPCG_NOHPX")
endif()

if(NOT HPCG_NOOPENMP)
    if(NOT HPCG_NOHPX)
      message(FATAL_ERROR "Enabling HPX and OpenMP at the same time is not supported")
    endif()

    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
else()
    add_definitions("-DHPCG_NOOPENMP")
endif()

if(NOT HPCG_NOMPI)
    if(NOT HPCG_NOHPX)
      message(FATAL_ERROR "Enabling HPX and MPI at the same time is not supported")
    endif()

    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
else()
    add_definitions("-DHPCG_NOMPI")
endif()

if(HPCG_DEBUG)
    add_definitions("-DHPCG_DEBUG")
endif()

add_subdirectory(src)

