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

#ifndef NVCV_PRIV_CORE_IMAGEBATCHMANAGER_HPP
#define NVCV_PRIV_CORE_IMAGEBATCHMANAGER_HPP

#include "IContext.hpp"
#include "ImageBatchVarShape.hpp"

namespace nv::cv::priv {

using ImageBatchManager = CoreObjManager<NVCVImageBatchHandle>;

using ImageBatchStorage = CompatibleStorage<ImageBatchVarShape>;

template<>
class CoreObjManager<NVCVImageBatchHandle> : public HandleManager<IImageBatch, ImageBatchStorage>
{
    using Base = HandleManager<IImageBatch, ImageBatchStorage>;

public:
    using Base::Base;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_IMAGEBATCHMANAGER_HPP
