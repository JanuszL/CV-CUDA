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
set(C_WARNING_FLAGS "-Wall -Wno-unknown-pragmas -Wpointer-arith -Wmissing-declarations -Wredundant-decls -Wmultichar -Wno-unused-local-typedefs -Wunused ")

# let the compiler help us marking virtual functions with override
set(CXX_WARNING_FLAGS "-Wsuggest-override")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror ${C_WARNING_FLAGS} ${CXX_WARNING_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror ${C_WARNING_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings ${C_WARNING_FLAGS} ${CXX_WARNING_FLAGS}")

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

if(ENABLE_SANITIZER)
    set(COMPILER_SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fsanitize=leak
        -fsanitize=undefined
        -fno-sanitize-recover=all
        # not properly supported, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64234
        #-static-libasan
        -static-liblsan
        -static-libubsan)
    string(REPLACE ";" " " COMPILER_SANITIZER_FLAGS "${COMPILER_SANITIZER_FLAGS}" )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILER_SANITIZER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMPILER_SANITIZER_FLAGS}")
endif()
