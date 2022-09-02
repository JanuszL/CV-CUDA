# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -ggdb")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -ggdb")

# Match warning setup with GVS
set(WARNING_FLAGS "-Wall -Werror -Wpointer-arith \
-Wmissing-declarations -Wredundant-decls -Wmultichar \
-Wno-unused-local-typedefs -Wunused -Wno-unknown-pragmas")

# let the compiler help us marking virtual functions with override
set(CXX_WARNING_FLAGS "${CXX_WARNING_FLAGS} -Wsuggest-override")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${CXX_WARNING_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    message(FATAL_ERROR "Must use gcc>=11.0 to compile CV-CUDA, you're using ${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(LTO_ENABLED ON)
else()
    set(LTO_ENABLED OFF)
endif()
