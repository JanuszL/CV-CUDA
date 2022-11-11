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

list(APPEND CPACK_COMPONENTS_ALL samples)

set(CPACK_COMPONENT_SAMPLES_DISABLED true)
set(CPACK_COMPONENT_SAMPLES_DISPLAY_NAME "Samples")
set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "NVIDIA CV-CUDA Samples")

if(UNIX)
    set(CVCUDA_SAMPLES_FILE_NAME "cvcuda-samples-${CVCUDA_VERSION_BUILD}")
    set(CPACK_DEBIAN_SAMPLES_FILE_NAME "${CVCUDA_SAMPLES_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_SAMPLES_FILE_NAME "${CVCUDA_SAMPLES_FILE_NAME}")

    set(CPACK_DEBIAN_SAMPLES_PACKAGE_NAME "cvcuda${PROJECT_VERSION_MAJOR}-samples")

    set(CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS "${CPACK_DEBIAN_DEV_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")
else()
    set(CPACK_COMPONENT_SAMPLES_DEPENDS dev)
endif()
