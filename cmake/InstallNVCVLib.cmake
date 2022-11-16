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

list(APPEND CPACK_COMPONENTS_ALL lib)
set(CPACK_COMPONENT_LIB_DISPLAY_NAME "Runtime libraries")
set(CPACK_COMPONENT_LIB_DESCRIPTION "NVIDIA NVCV library")
set(CPACK_COMPONENT_LIB_REQUIRED true)

set(NVCV_PACKAGE_NAME "nvcv${NVCV_VERSION_MAJOR}")

if(UNIX)
    set(NVCV_LIB_FILE_NAME "nvcv-lib-${NVCV_VERSION_BUILD}")

    set(CPACK_DEBIAN_LIB_FILE_NAME "${NVCV_LIB_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_LIB_FILE_NAME "${NVCV_LIB_FILE_NAME}")

    configure_file(cpack/debian_lib_postinst.in cpack/lib/postinst @ONLY)
    configure_file(cpack/debian_lib_prerm.in cpack/lib/prerm @ONLY)

    set(CPACK_DEBIAN_LIB_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/prerm")

    # as per debian convention, use the library file name
    set(CPACK_DEBIAN_LIB_PACKAGE_NAME "lib${NVCV_PACKAGE_NAME}")

    set(CPACK_DEBIAN_LIB_PACKAGE_DEPENDS "libstdc++6, libc6")

    if(ENABLE_SANITIZER)
        set(CPACK_DEBIAN_LIB_PACKAGE_DEPENDS "${CPACK_DEBIAN_LIB_PACKAGE_DEPENDS}, libasan6")
    endif()

    configure_file(cpack/ld.so.conf.in cpack/ld.so.conf @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpack/ld.so.conf
        DESTINATION "etc/ld.so.conf.d"
        RENAME ${CPACK_PACKAGE_NAME}.conf
        COMPONENT lib)
endif()

# Handle licenses, they go together with the library
install(FILES ${CPACK_RESOURCE_FILE_LICENSE}
    DESTINATION doc
    RENAME CVCUDA_EULA.txt
    COMPONENT lib)
