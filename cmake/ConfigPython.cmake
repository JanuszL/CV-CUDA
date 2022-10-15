# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

if(ENABLE_SANITIZERS)
    message(FATAL_ERROR "NVCV python modules don't work on sanitized builds")
endif()

# Because we python as subproject, we need to create a fake Findnvcv.cmake so
# that find_package will find our local nvcv library and headers
set(FINDNVCV_CONTENTS
[=[
add_library(nvcv SHARED IMPORTED)
target_include_directories(nvcv INTERFACE "$<TARGET_PROPERTY:nvcv,INTERFACE_INCLUDE_DIRECTORIES>")
]=])

if(CMAKE_CONFIGURATION_TYPES)
    set(NVCV_CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES})
else()
    set(NVCV_CONFIG_TYPES ${CMAKE_BUILD_TYPE})
endif()

foreach(cfg ${NVCV_CONFIG_TYPES})
    string(TOLOWER ${cfg} cfg_lower)
    set(FINDNVCV_CONTENTS
"${FINDNVCV_CONTENTS}include(nvcv_${cfg_lower})
")
endforeach()

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/Findnvcv.cmake CONTENT "${FINDNVCV_CONTENTS}")

list(LENGTH "${NVCV_CONFIG_TYPES}" num_configs)
if(${num_configs} EQUAL 1)
    set(NVCV_BUILD_SUFFIX "")
else()
    set(NVCV_BUILD_SUFFIX "_$<UPPER_CASE:$<CONFIG>>")
endif()

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcv_$<LOWER_CASE:$<CONFIG>>.cmake CONTENT
"set_target_properties(nvcv PROPERTIES IMPORTED_LOCATION${NVCV_BUILD_SUFFIX} \"$<TARGET_FILE:nvcv>\"
                                       IMPORTED_IMPLIB${NVCV_BUILD_SUFFIX} \"$<TARGET_LINKER_FILE:nvcv>\")
")

# Python versions to build already set?
if(PYTHON_VERSIONS)
    set(USE_DEFAULT_PYTHON false)
# If not, gets the default version from FindPython
else()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(PYTHON_VERSIONS ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
    set(USE_DEFAULT_PYTHON true)
endif()
