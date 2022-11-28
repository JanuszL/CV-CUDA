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

#ifndef NVCV_PRIV_CORE_CONTEXT_HPP
#define NVCV_PRIV_CORE_CONTEXT_HPP

#include "AllocatorManager.hpp"
#include "DefaultAllocator.hpp"
#include "IContext.hpp"
#include "ImageBatchManager.hpp"
#include "ImageManager.hpp"
#include "TensorManager.hpp"

namespace nv::cv::priv {

class Context final : public IContext
{
public:
    Context();
    ~Context();

    CoreObjManager<NVCVImageHandle>      &imageManager() override;
    CoreObjManager<NVCVImageBatchHandle> &imageBatchManager() override;
    CoreObjManager<NVCVTensorHandle>     &tensorManager() override;
    CoreObjManager<NVCVAllocatorHandle>  &allocatorManager() override;

    IAllocator &allocDefault() override;

private:
    // Order is important due to inter-dependencies
    DefaultAllocator  m_allocDefault;
    AllocatorManager  m_allocatorManager;
    ImageManager      m_imageManager;
    ImageBatchManager m_imageBatchManager;
    TensorManager     m_tensorManager;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_CONTEXT_HPP
