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

add_library(cvcuda_legacy STATIC
    ./operators/custom_crop.cu
    ./operators/reformat.cu
    ./operators/resize.cu
    ./operators/pad_and_stack.cu
    ./operators/convert_to.cu
    ./operators/normalize.cu
    CvCudaLegacyHelpers.cpp
)

target_include_directories(cvcuda_legacy
    PUBLIC
        ../..
        ../../include
)

target_link_libraries(cvcuda_legacy
    PUBLIC
        nvcv_headers
        nvcv_private_core
        nvcv_format
        CUDA::cudart_static
)
