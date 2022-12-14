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

#ifndef NVCV_ITENSORDATA_HPP
#define NVCV_ITENSORDATA_HPP

#include "PixelType.hpp"
#include "TensorData.h"
#include "TensorShape.hpp"
#include "detail/Optional.hpp"

namespace nv { namespace cv {

// Interface hierarchy of tensor contents
class ITensorData
{
public:
    virtual ~ITensorData() = 0;

    int                         ndim() const;
    const TensorShape          &shape() const;
    const TensorShape::DimType &shape(int d) const;

    const TensorLayout &layout() const;

    PixelType dtype() const;

    const NVCVTensorData &cdata() const;

protected:
    ITensorData() = default;
    ITensorData(const NVCVTensorData &data);

    NVCVTensorData &cdata();

private:
    NVCVTensorData                        m_data;
    mutable detail::Optional<TensorShape> m_cacheShape;
};

class ITensorDataPitch : public ITensorData
{
public:
    virtual ~ITensorDataPitch() = 0;

    void *data() const;

    const int64_t &pitchBytes(int d) const;

protected:
    using ITensorData::ITensorData;
};

class ITensorDataPitchDevice : public ITensorDataPitch
{
public:
    virtual ~ITensorDataPitchDevice() = 0;

protected:
    using ITensorDataPitch::ITensorDataPitch;
};

}} // namespace nv::cv

#include "detail/ITensorDataImpl.hpp"

#endif // NVCV_ITENSORDATA_HPP
