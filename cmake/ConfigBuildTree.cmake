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

set(PROJECT_VERSION "${PROJECT_VERSION}${PROJECT_VERSION_SUFFIX}")
set(cvcuda_VERSION_SUFFIX "${PROJECT_VERSION_SUFFIX}")
set(cvcuda_VERSION "${PROJECT_VERSION}")

set(cvcuda_API_VERSION "${cvcuda_VERSION_MAJOR}.${cvcuda_VERSION_MINOR}")
math(EXPR cvcuda_API_CODE "${cvcuda_VERSION_MAJOR}*100 + ${cvcuda_VERSION_MINOR}")

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
