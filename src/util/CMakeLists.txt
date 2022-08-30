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

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL COMPONENTS Crypto REQUIRED)

add_library(nvcv_util
    Assert.cpp
    HashMD5.cpp
    Exception.cpp
    Status.cpp
    CheckError.cpp
    String.cpp
)

target_include_directories(nvcv_util
    INTERFACE
        ..
)

target_link_libraries(nvcv_util
    PUBLIC
        nvcv_headers
        CUDA::cudart_static
    PRIVATE
        OpenSSL::Crypto
)

target_compile_definitions(nvcv_util
    PUBLIC
        # Must always be enabled so that we can check assertion
        # failures in release mode.
        -DNVCV_DEBUG=1
)

if(EXPOSE_CODE)
    target_compile_definitions(nvcv_util
        PUBLIC
            -DNVCV_EXPOSE_CODE=1
    )
endif()
