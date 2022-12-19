# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(cvcuda_legacy STATIC
    ./operators/custom_crop.cu
    ./operators/reformat.cu
    ./operators/resize.cu
    ./operators/resize_var_shape.cu
    ./operators/convert_to.cu
    ./operators/normalize.cu
    ./operators/pad_and_stack.cu
    ./operators/normalize_var_shape.cu
    ./operators/copy_make_border.cu
    ./operators/copy_make_border_var_shape.cu
    ./operators/center_crop.cu
    ./operators/rotate.cu
    ./operators/rotate_var_shape.cu
    ./operators/filter.cu
    ./operators/erase.cu
    ./operators/erase_var_shape.cu
    ./operators/filter_var_shape.cu
    ./operators/median_blur.cu
    ./operators/bilateral_filter.cu
    ./operators/bilateral_filter_var_shape.cu
    ./operators/median_blur_var_shape.cu
    ./operators/morphology.cu
    ./operators/cvt_color.cu
    ./operators/warp.cu
    ./operators/warp_var_shape.cu
    ./operators/cvt_color_var_shape.cu
    ./operators/composite.cu
    ./operators/morphology_var_shape.cu
    ./operators/channel_reorder_var_shape.cu
    ./operators/flip.cu
    ./operators/flip_or_copy_var_shape.cu
    ./operators/composite_var_shape.cu
    ./operators/gamma_contrast_var_shape.cu
    ./operators/pillow_resize.cu
    ./operators/pillow_resize_var_shape.cu
    CvCudaLegacyHelpers.cpp
)

target_include_directories(cvcuda_legacy
    PUBLIC
        ../..
        ../../include
)

target_link_libraries(cvcuda_legacy
    PUBLIC
        nvcv_headers
        nvcv_private_core
        nvcv_format
        CUDA::cudart_static
)
