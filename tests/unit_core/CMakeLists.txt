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

add_executable(nvcv_test_unit_core
    ../Main.cpp

    TestHashMD5.cpp
    TestValueList.cpp
    TestValueTests.cpp
    TestTypeList.cpp
    TestTypedTests.cpp
    TestCheckError.cpp
    TestString.cpp
    TestStaticVector.cpp
    TestVersion.cpp
    TestIndexSequence.cpp
    TestMath.cpp
    TestMetaprogramming.cpp
    TestOptional.cpp
    TestLegacyHelpers.cpp
    TestLockFreeStack.cpp
    TestHandleManager.cpp
)

target_compile_definitions(nvcv_test_unit_core
    PRIVATE
        -DNVCV_UNIT_TEST=1
)

target_link_libraries(nvcv_test_unit_core
    PRIVATE
        nvcv_util
        nvcv_test_common
        nvcv_private_core
        nvcv_private_operators
)

nvcv_add_test(nvcv_test_unit_core)
