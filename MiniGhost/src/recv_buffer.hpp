//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_RECV_BUFFER_HPP
#define HPX_EXAMPLES_MINI_GHOST_RECV_BUFFER_HPP

#include <grid.hpp>
#include <unpack_buffer.hpp>

#include <hpx/lcos/async.hpp>
#include <hpx/lcos/local/receive_buffer.hpp>
#include <hpx/util/serialize_buffer.hpp>

namespace mini_ghost {
    template <typename BufferType, std::size_t Zone>
    struct recv_buffer
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(recv_buffer);
    public:
        typedef hpx::lcos::local::spinlock mutex_type;

        typedef typename BufferType::value_type value_type;

        typedef
            BufferType
            buffer_type;

        recv_buffer()
          : valid_(false)
        {}

        recv_buffer(recv_buffer && other)
          : buffer_(std::move(other.buffer_))
          , valid_(other.valid_)
        {
        }

        recv_buffer& operator=(recv_buffer &&other)
        {
            if(this != &other)
            {
                buffer_     = std::move(other.buffer_);
                valid_      = other.valid_;
            }
            return *this;
        }

        ~recv_buffer()
        {
        }

        void operator()(grid<value_type> & g, std::size_t step)
        {
            HPX_ASSERT(valid_);
            hpx::util::high_resolution_timer timer;
            buffer_type buffer = buffer_.receive(step).get();
            double elapsed = timer.elapsed();
            profiling::data().time_wait(elapsed);
            switch(Zone)
            {
                case EAST:
                    profiling::data().time_wait_x(elapsed);
                    break;
                case WEST:
                    profiling::data().time_wait_x(elapsed);
                    break;
                case NORTH:
                    profiling::data().time_wait_y(elapsed);
                    break;
                case SOUTH:
                    profiling::data().time_wait_y(elapsed);
                    break;
                case FRONT:
                    profiling::data().time_wait_z(elapsed);
                    break;
                case BACK:
                    profiling::data().time_wait_z(elapsed);
                    break;
            }
            unpack_buffer<Zone>::call(g, buffer);
        }

        void set_buffer(buffer_type buffer, std::size_t step)
        {
            HPX_ASSERT(valid_);
            buffer_.store_received(step, std::move(buffer));
        }

        hpx::lcos::local::receive_buffer<buffer_type, mutex_type> buffer_;
        bool valid_;
    };
}

#endif
