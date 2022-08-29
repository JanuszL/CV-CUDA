# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Global configuration =========================================

if(UNIX)
    set(CPACK_SYSTEM_NAME "x86_64-linux")
else()
    message(FATAL_ERROR "Architecture not supported")
endif()

set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_CONTACT "CV-CUDA Support <cv-cuda@exchange.nvidia.com>")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://confluence.nvidia.com/display/CVCUDA")

# ARCHIVE installer doesn't work with absolute install destination
# we have to error out in this case
set(CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)

set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_TWEAK "${PROJECT_VERSION_TWEAK}")
set(CPACK_PACKAGE_VERSION_SUFFIX "${PROJECT_VERSION_SUFFIX}")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.cvcuda")
set(CPACK_MONOLITHIC_INSTALL OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CPACK_STRIP_FILES false)
else()
    set(CPACK_STRIP_FILES true)
endif()

set(CPACK_VERBATIM_VARIABLES true)
set(CPACK_GENERATOR TXZ)
set(CPACK_THREADS 0) # use all cores
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# we split the file name components with '-', so the version component can't
# have this character, let's replace it by '_'
string(REPLACE "-" "_" tmp ${CPACK_PACKAGE_VERSION})
set(PACKAGE_FULL_VERSION "${tmp}-cuda${CUDA_VERSION_MAJOR}-${CPACK_SYSTEM_NAME}")

set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PACKAGE_FULL_VERSION}")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}${PROJECT_VERSION_MAJOR}")

set(nvcv_PACKAGE_FILE_NAME "nvcv-${PACKAGE_FULL_VERSION}")
set(nvcv_PACKAGE_NAME "nvcv${PROJECT_VERSION_MAJOR}")

# CI needs this VERSION file to select the correct installer packages
add_custom_target(cvcuda_version_file ALL
        COMMAND ${CMAKE_COMMAND} -E echo ${PACKAGE_FULL_VERSION} > ${cvcuda_BINARY_DIR}/VERSION)

if(UNIX)
    set(CPACK_GENERATOR ${CPACK_GENERATOR} DEB)

    set(CPACK_COMPONENTS_GROUPING IGNORE)

    # Debian options ----------------------------------------
    set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS true)
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION ON)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS off)
    set(CPACK_DEBIAN_COMPRESSION_TYPE xz)

    # Create several .debs, one for each component
    set(CPACK_DEB_COMPONENT_INSTALL ON)

    # Archive options -----------------------------------
    set(CPACK_ARCHIVE_THREADS 0) # use all cores
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
endif()

set(CPACK_COMPONENTS_ALL "")

# Runtime libraries =========================================

list(APPEND CPACK_COMPONENTS_ALL lib)
set(CPACK_COMPONENT_LIB_DISPLAY_NAME "Runtime libraries")
set(CPACK_COMPONENT_LIB_DESCRIPTION "NVIDIA NVCV library")
set(CPACK_COMPONENT_LIB_REQUIRED true)

if(UNIX)
    set(CVCUDA_LIB_FILE_NAME "nvcv-lib-${PACKAGE_FULL_VERSION}")

    set(CPACK_DEBIAN_LIB_FILE_NAME "${CVCUDA_LIB_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_LIB_FILE_NAME "${CVCUDA_LIB_FILE_NAME}")

    configure_file(cpack/debian_lib_postinst.in cpack/lib/postinst @ONLY)
    configure_file(cpack/debian_lib_prerm.in cpack/lib/prerm @ONLY)

    set(CPACK_DEBIAN_LIB_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/prerm")

    # as per debian convention, use the library file name
    set(CPACK_DEBIAN_LIB_PACKAGE_NAME "lib${nvcv_PACKAGE_NAME}")

    set(CPACK_DEBIAN_LIB_PACKAGE_DEPENDS "libstdc++6, libc6")

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

math(EXPR CVCUDA_NEXT_VERSION_MINOR "${cvcuda_VERSION_MINOR}+1")
set(CVCUDA_NEXT_API_VERSION "${cvcuda_VERSION_MAJOR}.${CVCUDA_NEXT_VERSION_MINOR}")

# Development =================================================
list(APPEND CPACK_COMPONENTS_ALL dev)

set(CPACK_COMPONENT_DEV_DISPLAY_NAME "Development")
set(CPACK_COMPONENT_DEV_DESCRIPTION "NVIDIA CV-CUDA C/C++ development library and headers")

if(UNIX)
    set(nvcv_DEV_FILE_NAME "nvcv-dev-${PACKAGE_FULL_VERSION}")

    set(CPACK_DEBIAN_DEV_FILE_NAME "${nvcv_DEV_FILE_NAME}.deb")
    set(CPACK_ARCHIVE_DEV_FILE_NAME "${nvcv_DEV_FILE_NAME}")

    # dev package works with any current and futures ABIs, provided major version
    # is the same
    set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "lib${nvcv_PACKAGE_NAME} (>= ${cvcuda_API_VERSION})")

    set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${nvcv_PACKAGE_NAME}-dev")

    # We're not adding compiler and cmake as dependencies, users can choose
    # whatever toolchain they want.

    # Set up control files
    set(CVCUDA_USR_LIB_DIR /usr/lib)

    set(args -DCVCUDA_SOURCE_DIR=${CMAKE_SOURCE_DIR}
             -DCVCUDA_BINARY_DIR=${CMAKE_BINARY_DIR}
             -Dnvcv_LIB_LINKER_FILE_NAME=$<TARGET_LINKER_FILE_NAME:nvcv>)

    foreach(var CMAKE_INSTALL_PREFIX
                CMAKE_INSTALL_INCLUDEDIR
                CMAKE_INSTALL_LIBDIR
                nvcv_PACKAGE_NAME
                CMAKE_LIBRARY_ARCHITECTURE
                cvcuda_API_CODE
                CVCUDA_USR_LIB_DIR)

        list(APPEND args "-D${var}=${${var}}")
    endforeach()

    add_custom_target(nvcv_dev_control_extra ALL
        COMMAND cmake ${args} -DSOURCE=${CMAKE_SOURCE_DIR}/cpack/debian_dev_prerm.in -DDEST=cpack/dev/prerm -P ${CMAKE_SOURCE_DIR}/cpack/ConfigureFile.cmake
        COMMAND cmake ${args} -DSOURCE=${CMAKE_SOURCE_DIR}/cpack/debian_dev_postinst.in -DDEST=cpack/dev/postinst -P ${CMAKE_SOURCE_DIR}/cpack/ConfigureFile.cmake
        BYPRODUCTS cpack/dev/prerm cpack/dev/postinst
        DEPENDS cpack/debian_dev_prerm.in cpack/debian_dev_postinst.in
        VERBATIM)

    set(CPACK_DEBIAN_DEV_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/dev/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/dev/prerm")
else()
    set(CPACK_COMPONENT_DEV_DEPENDS lib)
endif()

# Test suite =================================================
if(BUILD_TESTS)
    list(APPEND CPACK_COMPONENTS_ALL tests)

    set(CPACK_COMPONENT_TESTS_DISABLED true)
    set(CPACK_COMPONENT_TESTS_DISPLAY_NAME "Tests")
    set(CPACK_COMPONENT_TESTS_DESCRIPTION "NVIDIA CV-CUDA test suite (internal use only)")
    set(CPACK_COMPONENT_TESTS_GROUP internal)

    if(UNIX)
        # Depend on current or any future ABI with same major version
        set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "lib${nvcv_PACKAGE_NAME} (>= ${cvcuda_API_VERSION})")

        set(CVCUDA_TESTS_FILE_NAME "cvcuda-tests-${PACKAGE_FULL_VERSION}")

        set(CPACK_DEBIAN_TESTS_FILE_NAME "${CVCUDA_TESTS_FILE_NAME}.deb")
        set(CPACK_ARCHIVE_TESTS_FILE_NAME "${CVCUDA_TESTS_FILE_NAME}")

    else()
        set(CPACK_COMPONENT_TESTS_DEPENDS lib)
    endif()
endif()

# Finish GPack configuration =================================================

include(CPack)

cpack_add_component_group(internal DISPLAY_NAME Internal DESCRIPTION "Internal packages, do not distribute")
