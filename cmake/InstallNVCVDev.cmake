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

list(APPEND CPACK_COMPONENTS_ALL dev)

set(CPACK_COMPONENT_DEV_DISPLAY_NAME "Development")
set(CPACK_COMPONENT_DEV_DESCRIPTION "NVIDIA CV-CUDA C/C++ development library and headers")

if(UNIX)
    set(NVCV_DEV_FILE_NAME "nvcv-dev-${NVCV_VERSION_BUILD}")

    set(CPACK_DEBIAN_DEV_FILE_NAME "${NVCV_DEV_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_DEV_FILE_NAME "${NVCV_DEV_FILE_NAME}")

    # dev package works with any current and futures ABIs, provided major version
    # is the same
    set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_NAME} (>= ${NVCV_VERSION_API})")

    set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${NVCV_PACKAGE_NAME}-dev")

    # We're not adding compiler and cmake as dependencies, users can choose
    # whatever toolchain they want.

    # Set up control files
    set(CVCUDA_USR_LIB_DIR /usr/lib)

    set(args -DCVCUDA_SOURCE_DIR=${PROJECT_SOURCE_DIR}
             -DCVCUDA_BINARY_DIR=${PROJECT_BINARY_DIR}
             -DNVCV_LIB_LINKER_FILE_NAME=$<TARGET_LINKER_FILE_NAME:nvcv>)

    foreach(var CMAKE_INSTALL_PREFIX
                CMAKE_INSTALL_INCLUDEDIR
                CMAKE_INSTALL_LIBDIR
                NVCV_PACKAGE_NAME
                CMAKE_LIBRARY_ARCHITECTURE
                NVCV_VERSION_API_CODE
                CVCUDA_USR_LIB_DIR)

        list(APPEND args "-D${var}=${${var}}")
    endforeach()

    add_custom_target(nvcv_dev_control_extra ALL
        COMMAND cmake ${args} -DSOURCE=${PROJECT_SOURCE_DIR}/cpack/debian_dev_prerm.in -DDEST=cpack/dev/prerm -P ${PROJECT_SOURCE_DIR}/cpack/ConfigureFile.cmake
        COMMAND cmake ${args} -DSOURCE=${PROJECT_SOURCE_DIR}/cpack/debian_dev_postinst.in -DDEST=cpack/dev/postinst -P ${PROJECT_SOURCE_DIR}/cpack/ConfigureFile.cmake
        BYPRODUCTS cpack/dev/prerm cpack/dev/postinst
        DEPENDS cpack/debian_dev_prerm.in cpack/debian_dev_postinst.in
        VERBATIM)

    set(CPACK_DEBIAN_DEV_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/dev/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/dev/prerm")
else()
    set(CPACK_COMPONENT_DEV_DEPENDS lib)
endif()
