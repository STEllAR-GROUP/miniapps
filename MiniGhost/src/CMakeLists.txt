# Copyright (c) 2013-2014 Thoams Heller
# Copyright (c) 2013-2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add MiniGhost executable
add_definitions(-DHPX_LIMIT=10)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_hpx_executable(mini_ghost
    MODULE mini_ghost
    AUTOGLOB)
