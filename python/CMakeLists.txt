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

cmake_minimum_required(VERSION 3.22)

project(nvcv_python CXX C)

set(CMAKE_CXX_STANDARD 20)

set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
add_subdirectory(${PYBIND11_SOURCE_DIR} pybind11)

find_package(CUDAToolkit REQUIRED)
find_package(nvcv REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/util nvcv_util)

set(imgformat_defs ${CMAKE_CURRENT_BINARY_DIR}/NVCVPythonImageFormatDefs.inc)
set(imgformat_script ${CMAKE_CURRENT_SOURCE_DIR}/gen_imgformat_list.sh)
set(imgformat_header ${CMAKE_CURRENT_SOURCE_DIR}/../src/include/nvcv/ImageFormat.h)
add_custom_command(OUTPUT ${imgformat_defs}
    COMMAND ${imgformat_script} "${imgformat_header}" > ${imgformat_defs}
    DEPENDS ${imgformat_script} ${imgformat_header}
)

set(pixtype_defs ${CMAKE_CURRENT_BINARY_DIR}/NVCVPythonPixelTypeDefs.inc)
set(pixtype_script ${CMAKE_CURRENT_SOURCE_DIR}/gen_pixtype_list.sh)
set(pixtype_header ${CMAKE_CURRENT_SOURCE_DIR}/../src/include/nvcv/PixelType.h)
add_custom_command(OUTPUT ${pixtype_defs}
    COMMAND ${pixtype_script} "${pixtype_header}" > ${pixtype_defs}
    DEPENDS ${pixtype_script} ${pixtype_header}
)

pybind11_add_module(nvcv_python MODULE
    ${imgformat_defs}
    ${pixtype_defs}
    Assert.cpp
    Cache.cpp
    CheckError.cpp
    PyUtil.cpp
    String.cpp
    Main.cpp
    ImageFormat.cpp
    PixelType.cpp
    BorderType.cpp
    Object.cpp
    Stream.cpp
    StreamStack.cpp
    Resource.cpp
    ResourceGuard.cpp
    Container.cpp
    Tensor.cpp
    Image.cpp
    ImageBatch.cpp
    CudaBuffer.cpp
    InterpolationType.cpp
    Rect.cpp
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
)

target_include_directories(nvcv_python
    PRIVATE
        # for imgformat_defs
        ${CMAKE_CURRENT_BINARY_DIR}
)

set_target_properties(nvcv_python PROPERTIES
    OUTPUT_NAME nvcv
    DEBUG_POSTFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

target_link_libraries(nvcv_python
    PRIVATE
        nvcv
        nvcv_util_compat
        CUDA::cudart_static
        -static-libstdc++
)

target_compile_options(nvcv_python
    PRIVATE
        -Wno-deprecated-declarations
        -Wsuggest-override
        -Wall -Werror -Wpointer-arith
        -Wmissing-declarations -Wredundant-decls -Wmultichar
        -Wno-unused-local-typedefs -Wunused -Wno-unknown-pragmas
)

include(GNUInstallDirs)

install(TARGETS nvcv_python
    LIBRARY DESTINATION lib
)
