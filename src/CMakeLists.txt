# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_library(nvcv SHARED Foo.cu)

target_include_directories(nvcv
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Config for semver 2.0
set_target_properties(nvcv PROPERTIES
    VERSION "${cvcuda_VERSION}"
    SOVERSION "${cvcuda_VERSION_MAJOR}"
)

# Configure symbol visibility
set_target_properties(nvcv PROPERTIES VISIBILITY_INLINES_HIDDEN on
                                        C_VISIBILITY_PRESET hidden
                                        CXX_VISIBILITY_PRESET hidden
                                        CUDA_VISIBILITY_PRESET hidden)
target_compile_definitions(nvcv PRIVATE -DCUPVA_EXPORTING=1)

# Reduce executable size ==========================

# Configure the library linker to remove unused code
target_link_options(nvcv PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,--gc-sections -Wl,--as-needed)
# Put each function and it's data into separate linker sections
target_compile_options(nvcv PRIVATE -ffunction-sections -fdata-sections)

# Installer

install(TARGETS nvcv
        EXPORT nvcv
        COMPONENT lib
        LIBRARY NAMELINK_COMPONENT dev)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(nvcv-config-version.cmake
                                 COMPATIBILITY SameMajorVersion)

install(DIRECTORY include/nvcv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev)

if(UNIX)
    install(EXPORT nvcv
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nvcv"
        FILE nvcv-config.cmake
        COMPONENT dev)

    # WAR for https://gitlab.kitware.com/cmake/cmake/-/issues/23563
    install(CODE "set(CVCUDA_CONFIG_PATH \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/nvcv/nvcv-config.cmake\")"
        COMPONENT dev)

    install(CODE [=[
        file(READ "${CVCUDA_CONFIG_PATH}" contents)
        string(REPLACE "get_filename_component(_IMPORT_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)"
[[
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" REALPATH)
]]
            contents "${contents}")
        file(WRITE "${CVCUDA_CONFIG_PATH}" "${contents}")
        ]=]
        COMPONENT dev)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nvcv-config-version.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nvcv"
        COMPONENT dev)
endif()
