/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_TESTS_DEVICE_BORDER_VAR_SHAPE_WRAP_HPP
#define NVCV_TESTS_DEVICE_BORDER_VAR_SHAPE_WRAP_HPP

#include <cuda_runtime.h> // for cudaStream_t, etc.

template<class DstWrapper, class SrcWrapper>
void DeviceRunFillBorderVarShape(DstWrapper &dstWrap, SrcWrapper &srcWrap, int3 dstMaxSize, int2 borderSize,
                                 cudaStream_t &stream);

template<class DstWrapper, class SrcWrapper>
void DeviceRunFillBorderVarShapeNHWC(DstWrapper &dstWrap, SrcWrapper &srcWrap, int3 dstMaxSize, int2 borderSize,
                                     int numChannels, cudaStream_t &stream);

#endif // NVCV_TESTS_DEVICE_BORDER_VAR_SHAPE_WRAP_HPP
