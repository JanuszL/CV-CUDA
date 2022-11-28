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

#include "Context.hpp"

#include "HandleManagerImpl.hpp"

#include <util/Assert.h>

namespace nv::cv::priv {

IContext &GlobalContext()
{
    static Context g_ctx;
    return g_ctx;
}

Context::Context()
    : m_allocatorManager("Allocator")
    , m_imageManager("Image")
    , m_imageBatchManager("ImageBatch")
    , m_tensorManager("Tensor")
{
}

Context::~Context()
{
    // empty
}

IAllocator &Context::allocDefault()
{
    return m_allocDefault;
}

ImageManager &Context::imageManager()
{
    return m_imageManager;
}

ImageBatchManager &Context::imageBatchManager()
{
    return m_imageBatchManager;
}

TensorManager &Context::tensorManager()
{
    return m_tensorManager;
}

AllocatorManager &Context::allocatorManager()
{
    return m_allocatorManager;
}

template class HandleManager<IImage, ImageStorage>;
template class HandleManager<IImageBatch, ImageBatchStorage>;
template class HandleManager<ITensor, TensorStorage>;
template class HandleManager<IAllocator, AllocatorStorage>;

} // namespace nv::cv::priv
