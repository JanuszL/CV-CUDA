/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TENSORDATA_HPP
#define NVCV_TENSORDATA_HPP

#include "DataType.hpp"
#include "ITensorData.hpp"
#include "TensorShape.hpp"

namespace nv { namespace cv {

// TensorDataStridedDevice definition -----------------------

class TensorDataStridedDevice : public ITensorDataStridedDevice
{
public:
    using Buffer = NVCVTensorBufferStrided;

    explicit TensorDataStridedDevice(const TensorShape &shape, const DataType &dtype, const Buffer &data);
    explicit TensorDataStridedDevice(const NVCVTensorData &data);
};

}} // namespace nv::cv

#include "detail/TensorDataImpl.hpp"

#endif // NVCV_TENSORDATA_HPP
