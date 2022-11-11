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

list(APPEND CPACK_COMPONENTS_ALL tests)

set(CPACK_COMPONENT_TESTS_DISABLED true)
set(CPACK_COMPONENT_TESTS_DISPLAY_NAME "Tests")
set(CPACK_COMPONENT_TESTS_DESCRIPTION "NVIDIA CV-CUDA test suite (internal use only)")
set(CPACK_COMPONENT_TESTS_GROUP internal)

if(UNIX)
    # Depend on current or any future ABI with same major version
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")
    # External dependencies
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS},libssl3")

    set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "cvcuda${PROJECT_VERSION_MAJOR}-tests")

    set(CVCUDA_TESTS_FILE_NAME "cvcuda-tests-${CVCUDA_VERSION_BUILD}")

    set(CPACK_DEBIAN_TESTS_FILE_NAME "${CVCUDA_TESTS_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_TESTS_FILE_NAME "${CVCUDA_TESTS_FILE_NAME}")

else()
    set(CPACK_COMPONENT_TESTS_DEPENDS lib)
endif()
