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

# sanitized builds of our python module don't work,
# python interpreter exits with errorcode 1, nothing else
# is output
if(ENABLE_SANITIZERS)
    return()
endif()

file(GLOB SOURCES CONFIGURE_DEPENDS "*.py")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MODULE_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# create the test script for our build tree
set(PYTHON_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
string(REPLACE ";" " " PYTHON_TEST_VERSIONS "${PYTHON_VERSIONS}")

set(PYTHON_MODULE_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
configure_file(nvcv_test_python.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/nvcv_test_python @ONLY)

set(PYTHON_TEST_INSTDIR "${CMAKE_INSTALL_DATADIR}/python")

install(FILES ${SOURCES}
        DESTINATION "${PYTHON_TEST_INSTDIR}"
        COMPONENT tests)

# create the test script that will get installed
set(PYTHON_TEST_DIR ${CMAKE_INSTALL_PREFIX}/${PYTHON_TEST_INSTDIR})
set(PYTHON_MODULE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
configure_file(nvcv_test_python.in nvcv_test_python @ONLY)

nvcv_add_test(${CMAKE_CURRENT_BINARY_DIR}/nvcv_test_python)
