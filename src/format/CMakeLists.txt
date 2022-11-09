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

# nvcv format private implementation
add_subdirectory(priv)

# Making it an "object" library because it contains
# methods that are exported via libnvcv
add_library(nvcv_format OBJECT
    ColorSpec.cpp
    DataLayout.cpp
    DataType.cpp
    ImageFormat.cpp
)

target_include_directories(nvcv_format
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(nvcv_format
    PUBLIC
        nvcv_format_priv
        nvcv_headers
        nvcv_util
)

# target used when only public headers are needed
add_library(nvcv_format_headers INTERFACE)
target_include_directories(nvcv_format_headers INTERFACE include)

install(DIRECTORY include/nvcv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
