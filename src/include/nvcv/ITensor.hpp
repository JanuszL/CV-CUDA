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

#ifndef NVCV_ITENSOR_HPP
#define NVCV_ITENSOR_HPP

#include "PixelType.hpp"
#include "Tensor.h"
#include "TensorData.hpp"
#include "TensorLayout.hpp"
#include "TensorShape.hpp"
#include "detail/Optional.hpp"

namespace nv { namespace cv {

class ITensor
{
public:
    virtual ~ITensor() = default;

    NVCVTensorHandle handle() const;

    int          ndim() const;
    TensorShape  shape() const;
    PixelType    dtype() const;
    TensorLayout layout() const;

    const ITensorData *exportData() const;

    void  setUserPointer(void *ptr);
    void *userPointer() const;

private:
    virtual NVCVTensorHandle doGetHandle() const = 0;

    mutable detail::Optional<TensorDataPitchDevice> m_cacheData;
};

}} // namespace nv::cv

#include "detail/ITensorImpl.hpp"

#endif // NVCV_ITENSOR_HPP
