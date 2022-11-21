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

project(nvcv_test_system_cudatools)
set(CMAKE_FOLDER tests)

add_executable(nvcv_test_system_cudatools
    TestLinAlg.cpp
    TestTensorWrap.cpp
    DeviceTensorWrap.cu
    TestSaturateCast.cpp
    DeviceSaturateCast.cu
    TestRangeCast.cpp
    TestMathWrappers.cpp
    DeviceMathWrappers.cu
    TestMathOps.cpp
    TestStaticCast.cpp
    TestDropCast.cpp
    TestTypeTraits.cpp
    TestMetaprogramming.cpp
)

target_link_libraries(nvcv_test_system_cudatools
    PUBLIC
        nvcv_util
        nvcv_test_common
        nvcv
)

target_include_directories(nvcv_test_system_cudatools
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

nvcv_add_test(nvcv_test_system_cudatools)
