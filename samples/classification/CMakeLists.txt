# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

find_package(CUDA REQUIRED)

add_executable(nvcv_samples_classification Main.cpp)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-enum-enum-conversion")

target_link_libraries(nvcv_samples_classification nvcv CUDA::cudart_static nvcv_samples_common)

target_include_directories(nvcv_samples_classification
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

install(TARGETS nvcv_samples_classification
        EXPORT nvcv_samples_classification
        COMPONENT samples
        DESTINATION samples/bin)
