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

# nvcv operators private implementation
add_subdirectory(priv)

add_library(nvcv_operators SHARED
    Operator.cpp
    OpReformat.cpp
    OpResize.cpp
    OpCustomCrop.cpp
    OpNormalize.cpp
    OpPadAndStack.cpp
    OpConvertTo.cpp
    OpCopyMakeBorder.cpp
    OpCenterCrop.cpp
    OpRotate.cpp
    OpLaplacian.cpp
    OpGaussian.cpp
    OpErase.cpp
    OpAverageBlur.cpp
    OpConv2D.cpp
    OpMedianBlur.cpp
    OpBilateralFilter.cpp
    OpMorphology.cpp
    OpCvtColor.cpp
    OpWarpAffine.cpp
    OpWarpPerspective.cpp
    OpComposite.cpp
    OpChannelReorder.cpp
    OpFlip.cpp
    OpGammaContrast.cpp
    OpPillowResize.cpp
)

target_link_libraries(nvcv_operators
    PRIVATE
        nvcv_types
        nvcv_util
        nvcv_operators_priv
)

target_include_directories(nvcv_operators
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# target used when only public headers are needed
add_library(nvcv_operators_headers INTERFACE)
target_include_directories(nvcv_operators_headers INTERFACE include ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(nvcv_operators_headers INTERFACE nvcv_types_headers)

configure_version(nvcv_operators NVCV_OP nvcv/operators ${PROJECT_VERSION})
configure_symbol_versioning(nvcv_operators NVCV_OP nvcv_operators)
setup_dso(nvcv_operators ${PROJECT_VERSION})

target_compile_definitions(nvcv_operators PRIVATE -DNVCV_OP_EXPORTING=1)

# Installer

install(TARGETS nvcv_operators
        EXPORT nvcv_operators
        COMPONENT lib
        LIBRARY NAMELINK_COMPONENT dev)

install(DIRECTORY include/nvcv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
