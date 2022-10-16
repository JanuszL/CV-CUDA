# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(nvcv_samples_common LANGUAGES CXX)

add_library(nvcv_samples_common SHARED
                                TRTUtils.cpp
			        NvDecoder.cpp)
target_compile_options(nvcv_samples_common PRIVATE -Wno-deprecated-declarations -Wno-missing-declarations)
target_link_libraries(nvcv_samples_common CUDA::cudart_static TensorRT::nvinfer CUDA::nvjpeg)

install(TARGETS nvcv_samples_common
        EXPORT nvcv_samples_common
        COMPONENT samples
	DESTINATION samples/lib)
