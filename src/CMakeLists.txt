# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# We're creating a shared lib, all projects it depends on must have
# PIC enabled
set(CMAKE_POSITION_INDEPENDENT_CODE on)

if(LTO_ENABLED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION on)
endif()

# general utilities library
add_subdirectory(util)

# nvcv core private implementation
add_subdirectory(private)

add_library(nvcv SHARED
    Version.cpp
    Status.cpp
    ColorSpec.cpp
    DataLayout.cpp
    PixelType.cpp
    ImageFormat.cpp
    Allocator.cpp
    Requirements.cpp
    Image.cpp
)

target_link_libraries(nvcv PRIVATE
    nvcv_private_core
    nvcv_util
    nvcv_format
    -static-libstdc++
    -static-libgcc
)

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

# configure version header --------------------------------
if(PROJECT_VERSION_TWEAK)
    set(NVCV_VERSION_TWEAK ${PROJECT_VERSION_TWEAK})
else()
    set(NVCV_VERSION_TWEAK 0)
endif()

math(EXPR NVCV_VERSION "${PROJECT_VERSION_MAJOR} * 1000000 + ${PROJECT_VERSION_MINOR} * 10000 + ${PROJECT_VERSION_PATCH} * 100 + ${NVCV_VERSION_TWEAK}")

configure_file(include/nvcv/Version.h.in include/nvcv/Version.h @ONLY ESCAPE_QUOTES)

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC NVCV_COMMIT)

configure_file(include/nvcv/detail/VersionUtils.h.in include/nvcv/detail/VersionUtils.h @ONLY ESCAPE_QUOTES)

# So that the generated headers are found
target_include_directories(nvcv
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/nvcv/Version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nvcv
        COMPONENT dev)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/nvcv/detail/VersionUtils.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nvcv/detail
        COMPONENT dev)

# Create exports file for symbol versioning ---------------------------------
set(EXPORTS_OUTPUT "${cvcuda_BINARY_DIR}/exports.ldscript")
target_link_libraries(nvcv
    PRIVATE
    -Wl,--version-script ${EXPORTS_OUTPUT}
)
get_target_property(CVCUDA_SOURCES nvcv SOURCES)
set(GEN_EXPORTS_SCRIPT "${cvcuda_SOURCE_DIR}/cmake/CreateExportsFile.cmake")

add_custom_command(OUTPUT ${EXPORTS_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR="${cvcuda_SOURCE_DIR}/src"
                             -DSOURCES="${CVCUDA_SOURCES}"
                             -DOUTPUT=${EXPORTS_OUTPUT}
                             -P "${GEN_EXPORTS_SCRIPT}"
    DEPENDS ${GEN_EXPORTS_SCRIPT} ${CVCUDA_SOURCES})

add_custom_target(create_exports_file DEPENDS ${EXPORTS_OUTPUT})
add_dependencies(nvcv create_exports_file)

# Configure symbol visibility ---------------------------------------------
set_target_properties(nvcv PROPERTIES VISIBILITY_INLINES_HIDDEN on
                                        C_VISIBILITY_PRESET hidden
                                        CXX_VISIBILITY_PRESET hidden
                                        CUDA_VISIBILITY_PRESET hidden)
target_compile_definitions(nvcv PRIVATE -DCUPVA_EXPORTING=1)

# our public function definitions have a different name than
# what's in the headers due to the versioning scheme. This will
# load to a lot of spurious -Wmissing-declaration warnings. Let's
# silence them out.
target_compile_options(nvcv PRIVATE -Wno-missing-declarations)

# target used when only public headers are needed
add_library(nvcv_headers INTERFACE)
target_include_directories(nvcv_headers INTERFACE include ${CMAKE_CURRENT_BINARY_DIR}/include)

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

install(DIRECTORY include/nvcv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(nvcv-config-version.cmake
                                 COMPATIBILITY SameMajorVersion)

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
