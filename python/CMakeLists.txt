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

cmake_minimum_required(VERSION 3.22)

project(nvcv_python CXX C)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_WARNINGS
    -Wno-deprecated-declarations
    -Wsuggest-override
    -Wall -Werror -Wpointer-arith
    -Wmissing-declarations -Wredundant-decls -Wmultichar
    -Wno-unused-local-typedefs -Wunused -Wno-unknown-pragmas
)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${COMPILE_WARNINGS}")

set(CMAKE_POSITION_INDEPENDENT_CODE true)

set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
add_subdirectory(${PYBIND11_SOURCE_DIR} pybind11)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/util nvcv_util)

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(operators)
