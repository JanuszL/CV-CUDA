# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(cvcuda_tests)
set(CMAKE_FOLDER tests)

enable_testing()

find_package(GTest REQUIRED)

add_library(cvcuda_test_main Main.cpp)
target_link_libraries(cvcuda_test_main PUBLIC GTest::gtest)

# Now we want to create a script in the output directory that will run
# all tests we define.
if(UNIX)
    file(TO_NATIVE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/" TESTS_DRIVER_DIR)
    set(TESTS_DRIVER "${TESTS_DRIVER_DIR}/run_tests.sh")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh.in ${TESTS_DRIVER}
        @ONLY)

    macro(cvcuda_add_test)
        cmake_parse_arguments(ADDTEST "" "NAME;COMMAND" "" ${ARGN})

        add_test(NAME "${ADDTEST_NAME}" COMMAND "${ADDTEST_COMMAND}")

        # Timeouts are in seconds
        set(TIMEOUT_KILL 800)
        set(TIMEOUT_TERM 620)

        set_tests_properties(${ADDTEST_NAME} PROPERTIES TIMEOUT ${TIMEOUT_TERM})

        file(APPEND "${TESTS_DRIVER}" "run ${ADDTEST_NAME}\n")
    endmacro()


    install(PROGRAMS ${TESTS_DRIVER}
        TYPE BIN
        COMPONENT tests)
else()
    macro(cvcuda_add_test)
        add_test(${ARGV})
    endmacro()
endif()

# Create build tree

# System tests for public API
add_subdirectory(system)
