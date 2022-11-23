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

#ifndef NVCV_PRIV_CORE_ICONTEXT_HPP
#define NVCV_PRIV_CORE_ICONTEXT_HPP

#include "ICoreObject.hpp"

typedef struct NVCVImage      *NVCVImageHandle;
typedef struct NVCVImageBatch *NVCVImageBatchHandle;
typedef struct NVCVTensor     *NVCVTensorHandle;
typedef struct NVCVAllocator  *NVCVAllocatorHandle;

namespace nv::cv::priv {

class IAllocator;

class IContext
{
public:
    virtual IAllocator &allocDefault() = 0;

    virtual CoreObjManager<NVCVImageHandle>      &imageManager()      = 0;
    virtual CoreObjManager<NVCVImageBatchHandle> &imageBatchManager() = 0;
    virtual CoreObjManager<NVCVTensorHandle>     &tensorManager()     = 0;
    virtual CoreObjManager<NVCVAllocatorHandle>  &allocatorManager()  = 0;
};

// Defined in Context.cpp
IContext &GlobalContext();

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_ICONTEXT_HPP
