# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(nvcv_test_system_core)
set(CMAKE_FOLDER tests)

add_executable(nvcv_test_system_core
    dummy.c
)

target_link_libraries(nvcv_test_system_core
    PUBLIC
        nvcv_test_main
        nvcv
)

nvcv_add_test(nvcv_test_system_core)
