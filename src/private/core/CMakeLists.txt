# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES

# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_library(nvcv_private_core STATIC
    Context.cpp
    TLS.cpp
    Status.cpp
    Version.cpp
    CustomAllocator.cpp
    DefaultAllocator.cpp
    IAllocator.cpp
    Requirements.cpp
    Exception.cpp
    Image.cpp
    ImageBatchVarShape.cpp
    Tensor.cpp
    TensorWrapDataPitch.cpp
    TensorLayout.cpp
    TensorData.cpp
    TensorShape.cpp
)

target_include_directories(nvcv_private_core
    PUBLIC
        ../..
)

target_compile_definitions(nvcv_private_core PUBLIC -DNVCV_EXPORTING=1)

target_link_libraries(nvcv_private_core
    PUBLIC
        nvcv_headers
        nvcv_util_sanitizer
        nvcv_format
        CUDA::cudart_static
)
