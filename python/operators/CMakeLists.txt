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

find_package(nvcv_operators REQUIRED)

pybind11_add_module(nvcv_operators_python MODULE
    Main.cpp
    InterpolationType.cpp
    BorderType.cpp
    ColorConversionCode.cpp
    MorphologyType.cpp
    OpReformat.cpp
    OpResize.cpp
    OpCustomCrop.cpp
    OpNormalize.cpp
    OpConvertTo.cpp
    OpPadAndStack.cpp
    OpCopyMakeBorder.cpp
    OpRotate.cpp
    OpErase.cpp
    OpGaussian.cpp
    OpMedianBlur.cpp
    OpLaplacian.cpp
    OpAverageBlur.cpp
    OpConv2D.cpp
    OpBilateralFilter.cpp
    OpCenterCrop.cpp
    OpWarpAffine.cpp
    OpWarpPerspective.cpp
    OpChannelReorder.cpp
    OpMorphology.cpp
    OpFlip.cpp
    OpCvtColor.cpp
    OpComposite.cpp
    OpGammaContrast.cpp
    OpPillowResize.cpp
)

set_target_properties(nvcv_operators_python PROPERTIES
    OUTPUT_NAME nvcv_operators
    DEBUG_POSTFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

target_link_libraries(nvcv_operators_python
    PRIVATE
        nvcv_util_compat
        nvcv_python
        nvcv_operators
        nvcv_python_common
        -static-libstdc++
)

include(GNUInstallDirs)

install(TARGETS nvcv_operators_python
    LIBRARY DESTINATION lib
)
