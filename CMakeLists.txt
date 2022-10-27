# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.22)

project(cvcuda
        LANGUAGES C CXX CUDA
        VERSION 0.1.0
        DESCRIPTION "CUDA-accelerated Computer Vision algorithms"
)

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "")

# Allow ccvcuda libraries with different major versions to be
# installed in parallel
set(CMAKE_INSTALL_PREFIX "/opt/nvidia/cvcuda${PROJECT_VERSION_MAJOR}")

# Options to configure the build tree =======
option(BUILD_TESTS "Enable testsuite" OFF)
option(BUILD_PYTHON "Build python bindings" OFF)
option(ENABLE_SANITIZER "Enabled sanitized build" OFF)

# Configure build tree ======================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

include(ConfigBuildTree)
include(ConfigCompiler)
include(ConfigCUDA)
include(ConfigCCache)
if(BUILD_PYTHON)
    include(ConfigPython)
endif()

# Define the build tree ====================

add_subdirectory(src)

if(BUILD_PYTHON)
    include(BuildPython)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

# Must be done after build tree is defined
include(ConfigCPack)

# Print build tree configuration ===========

message(STATUS "")
include(PrintConfig)
