# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(ENABLE_SANITIZERS)
    message(FATAL_ERROR "NVCV python modules don't work on sanitized builds")
endif()

# Because we python as subproject, we need to create a fake Findnvcv.cmake so
# that find_package will find our local nvcv library and headers
set(FINDNVCV_CONTENTS
[=[
add_library(nvcv SHARED IMPORTED)
target_include_directories(nvcv
    INTERFACE
    "$<TARGET_PROPERTY:nvcv,INTERFACE_INCLUDE_DIRECTORIES>"
    "$<TARGET_PROPERTY:nvcv_format,INTERFACE_INCLUDE_DIRECTORIES>"
    "$<TARGET_PROPERTY:nvcv_optools,INTERFACE_INCLUDE_DIRECTORIES>"
)
]=])

set(FINDNVCV_OP_CONTENTS
[=[
add_library(nvcv_operators SHARED IMPORTED)
target_include_directories(nvcv_operators
    INTERFACE
    "$<TARGET_PROPERTY:nvcv_operators,INTERFACE_INCLUDE_DIRECTORIES>"
    "$<TARGET_PROPERTY:nvcv_optools,INTERFACE_INCLUDE_DIRECTORIES>"
)
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
    set(FINDNVCV_OP_CONTENTS
"${FINDNVCV_OP_CONTENTS}include(nvcv_operators_${cfg_lower})
")
endforeach()

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/Findnvcv.cmake CONTENT "${FINDNVCV_CONTENTS}")
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/Findnvcv_operators.cmake CONTENT "${FINDNVCV_OP_CONTENTS}")

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

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcv_operators_$<LOWER_CASE:$<CONFIG>>.cmake CONTENT
"set_target_properties(nvcv_operators PROPERTIES IMPORTED_LOCATION${NVCV_BUILD_SUFFIX} \"$<TARGET_FILE:nvcv_operators>\"
                                                 IMPORTED_IMPLIB${NVCV_BUILD_SUFFIX} \"$<TARGET_LINKER_FILE:nvcv_operators>\")
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
