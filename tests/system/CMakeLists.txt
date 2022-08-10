# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(cvcuda_test_system)
set(CMAKE_FOLDER tests)

add_executable(cvcuda_test_system
    FooTests.cpp
)

target_link_libraries(cvcuda_test_system
    PUBLIC
        cvcuda_test_main
        cvcuda
)

cvcuda_add_test(NAME system COMMAND cvcuda_test_system)

install(TARGETS cvcuda_test_system
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tests)

