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

# nvcv_test_common ================================
# Utilities used in unit and system tests. They don't
# call the public API functions.

add_library(nvcv_test_common STATIC
    Printers.cpp
    BorderUtils.cpp
    ConvUtils.cpp
    TensorDataUtils.cpp
)

target_include_directories(nvcv_test_common
    INTERFACE
        ..
)

target_link_libraries(nvcv_test_common
    INTERFACE
        nvcv_util
)

target_compile_definitions(nvcv_test_common
    INTERFACE
        -DNVCV_EXPOSE_CODE=1
)

target_link_libraries(nvcv_test_common
    PUBLIC
        nvcv_headers
        nvcv_util
        nvcv_util_compat
)

# nvcv_test_common_system ==============================
# Utilities for system tests, they call the public API functions.

add_library(nvcv_test_common_system STATIC
    ObjectBag.cpp
)

target_include_directories(nvcv_test_common_system
    INTERFACE
        ..
)

target_link_libraries(nvcv_test_common_system
    PUBLIC
        nvcv
        nvcv_test_common
)
