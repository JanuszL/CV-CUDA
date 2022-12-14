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
