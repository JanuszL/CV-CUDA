# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.22)
project(nvcv_samples LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(nvcv)

set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

set(CMAKE_FOLDER samples)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(ConfigTensorRT)

set(SAMPLES common
	    classification
	    cropandresize)

foreach(sample ${SAMPLES})
    add_subdirectory(${sample})
endforeach()

install(DIRECTORY assets
    COMPONENT samples
    DESTINATION samples)

install(DIRECTORY scripts
    COMPONENT samples
    DESTINATION samples)

install(FILES "CMakeLists.txt" "ConfigTensorRT.cmake" "README.md"
    COMPONENT samples
    DESTINATION samples)

install(DIRECTORY ${SAMPLES}
    COMPONENT samples
    DESTINATION samples)
