# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)
list(GET CUDA_VERSION_LIST 2 CUDA_VERSION_PATCH)

find_package(CUDAToolkit ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} REQUIRED)

# CUDA version requirement:
# - to use gcc-11 (11.7)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.7")
    message(FATAL_ERROR "Minimum CUDA version supported is 11.7")
endif()

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

# Compress kernels to generate smaller executables
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin=--compress-all")

if(NOT USE_CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")

    # All architectures we build sass for
    list(APPEND CMAKE_CUDA_ARCHITECTURES
         70-real # Volta  - gv100/Tesla
         75-real # Turing - tu10x/GeForce
         80-real # Ampere - ga100/Tesla
         86-real # Ampere - ga10x/GeForce
    )

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
        list(APPEND CMAKE_CUDA_ARCHITECTURES
            89-real # Ada    - ad102/GeForce
            90-real # Hopper - gh100/Tesla
        )
    endif()

    # Required compute capability:
    # * compute_70: fast fp16 support + PTX for forward compatibility
    list(APPEND CMAKE_CUDA_ARCHITECTURES 70-virtual)
endif()
