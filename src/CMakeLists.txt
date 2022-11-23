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
    Config.cpp
    Version.cpp
    Status.cpp
    ColorSpec.cpp
    DataLayout.cpp
    PixelType.cpp
    ImageFormat.cpp
    Allocator.cpp
    Requirements.cpp
    Image.cpp
    ImageBatch.cpp
    Tensor.cpp
    TensorShape.cpp
    TensorLayout.cpp
    operators/Operator.cpp
    operators/OpReformat.cpp
    operators/OpResize.cpp
    operators/OpCustomCrop.cpp
    operators/OpNormalize.cpp
    operators/OpPadAndStack.cpp
    operators/OpConvertTo.cpp
    operators/OpCopyMakeBorder.cpp
    operators/OpCenterCrop.cpp
    operators/OpRotate.cpp
    operators/OpLaplacian.cpp
    operators/OpGaussian.cpp
)

target_link_libraries(nvcv PRIVATE
    nvcv_private_core
    nvcv_private_operators
    nvcv_util
    nvcv_format
)

target_include_directories(nvcv
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

configure_version(nvcv NVCV nvcv ${PROJECT_VERSION})
configure_symbol_versioning(nvcv NVCV nvcv)
setup_dso(nvcv ${PROJECT_VERSION})

target_compile_definitions(nvcv PRIVATE -DNVCV_EXPORTING=1)

# target used when only public headers are needed
add_library(nvcv_headers INTERFACE)
target_include_directories(nvcv_headers INTERFACE include ${CMAKE_CURRENT_BINARY_DIR}/include)

# Installer

install(TARGETS nvcv
        EXPORT nvcv
        COMPONENT lib
        LIBRARY NAMELINK_COMPONENT dev)

install(DIRECTORY include/nvcv
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inc")

install(DIRECTORY include/operators
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
