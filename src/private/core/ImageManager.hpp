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

#ifndef NVCV_PRIV_CORE_IMAGEMANAGER_HPP
#define NVCV_PRIV_CORE_IMAGEMANAGER_HPP

#include "IContext.hpp"
#include "Image.hpp"

namespace nv::cv::priv {

using ImageManager = CoreObjManager<NVCVImageHandle>;

using ImageStorage = CompatibleStorage<Image, ImageWrapData>;

template<>
class CoreObjManager<NVCVImageHandle> : public HandleManager<IImage, ImageStorage>
{
    using Base = HandleManager<IImage, ImageStorage>;

public:
    using Base::Base;

    static ImageManager &Instance()
    {
        return GlobalContext().imageManager();
    }
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_IMAGEMANAGER_HPP
