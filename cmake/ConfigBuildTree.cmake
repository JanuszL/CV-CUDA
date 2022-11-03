# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(CMAKE_DEBUG_POSTFIX "_d")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(GNUInstallDirs)

set(CMAKE_INSTALL_LIBDIR "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

# Executables try to find libnvvpi library relative to themselves.
set(CMAKE_BUILD_RPATH_USE_ORIGIN true)

# Whether assert dumps expose code
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(DEFAULT_EXPOSE_CODE OFF)
else()
    set(DEFAULT_EXPOSE_CODE ON)
endif()

option(EXPOSE_CODE "Expose in resulting binaries parts of our code" ${DEFAULT_EXPOSE_CODE})

# Are we inside a git repo and it has submodules enabled?
if(EXISTS ${CMAKE_SOURCE_DIR}/.git AND EXISTS ${CMAKE_SOURCE_DIR}/.gitmodules)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/.git/modules)
        message(FATAL_ERROR "git submodules not initialized. Did you forget to run 'git submodule update --init'?")
    endif()
endif()

if(UNIX)
    set(CVCUDA_SYSTEM_NAME "x86_64-linux")
else()
    message(FATAL_ERROR "Architecture not supported")
endif()

set(CVCUDA_BUILD_SUFFIX "cuda${CUDA_VERSION_MAJOR}-${CVCUDA_SYSTEM_NAME}")
