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

# system core -------------------------------------------------
add_executable(nvcv_test_system_core
    TestVersion.cpp
    TestStatus.cpp
    TestSize.cpp
    TestColorSpec.cpp
    TestDataLayout.cpp
    TestImageFormat.cpp
    TestPixelType.cpp
    TestAllocator.cpp
    TestRequirements.cpp
    TestImage.cpp
    TestImageBatch.cpp
    TestTensor.cpp
    TestTensorLayout.cpp
    TestTensorLayoutInfo.cpp
    TestTensorShapeInfo.cpp
    TestTensorDataAccess.cpp
)

target_link_libraries(nvcv_test_system_core
    PUBLIC
        nvcv_test_main
        nvcv_test_common_system
        nvcv
)

nvcv_add_test(nvcv_test_system_core)

# system core version commit ----------------------------------

# Separate test for versioning API so that only one target depends on
# vpi/detail/VersionUtil.h, which gets updated whenever the commit changes.
# This avoids spurious recompilation of all projects.
add_executable(nvcv_test_system_core_version_commit
    TestVersionCommit.cpp
)

target_link_libraries(nvcv_test_system_core_version_commit
    PRIVATE
        nvcv_test_main
        nvcv_test_common
        nvcv
)

nvcv_add_test(nvcv_test_system_core_version_commit)
