# Copyright (c) 2013-2014 Thoams Heller
# Copyright (c) 2013-2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Find and include HPX settings
if(NOT HPX_ROOT)
  message(FATAL "HPX_ROOT not set, unable to find HPX!")
endif()

# Require a recent version of cmake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

# This project is C++ based.
project(MiniGhost)

# This adds the HPX cmake configuration directory to the search path.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)

# Instruct cmake to find the HPX settings
find_package(HPX)

include_directories(${HPX_INCLUDE_DIR})
link_directories(${HPX_LIBRARY_DIR})

# Adjust auxiliary build settings for HPX to work properly
if(NOT MSVC)
  set(HPX_RPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}${HPX_RPATH})
else()
  # enable solution folders for MSVC
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  if(HPX_NO_INSTALL)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/$(Configuration)" CACHE PATH
      "Prefix prepended to install directories." FORCE)
  endif()
endif()

add_subdirectory(src)

