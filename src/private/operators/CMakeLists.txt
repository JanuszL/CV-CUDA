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

add_library(nvcv_private_operators STATIC
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
)

target_include_directories(nvcv_private_operators
    PUBLIC
        ../..
)

target_link_libraries(nvcv_private_operators
    PUBLIC
        nvcv_headers
        nvcv_util_sanitizer
        nvcv_format
        cvcuda_legacy
        CUDA::cudart_static
)
