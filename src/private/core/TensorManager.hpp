/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NVCV_PRIV_CORE_TENSORMANAGER_HPP
#define NVCV_PRIV_CORE_TENSORMANAGER_HPP

#include "IContext.hpp"
#include "Tensor.hpp"
#include "TensorWrapDataPitch.hpp"

#include <variant>

namespace nv::cv::priv {

using TensorManager = CoreObjManager<NVCVTensorHandle>;

using TensorStorage = std::variant<Tensor, TensorWrapDataPitch>;

template<>
class CoreObjManager<NVCVTensorHandle> : public HandleManager<ITensor, TensorStorage>
{
    using Base = HandleManager<ITensor, TensorStorage>;

public:
    using Base::Base;

    static TensorManager &Instance()
    {
        return GlobalContext().tensorManager();
    }
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_TENSORMANAGER_HPP
