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

set(CMAKE_FOLDER 3rdparty)

# disable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

set(CMAKE_POSITION_INDEPENDENT_CODE on)

# GoogleTest -----------------------------
set(INSTALL_GTEST off)
set(BUILD_GMOCK on)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(googletest)

# pybind11 -----------------------------
set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind11" PARENT_SCOPE)
