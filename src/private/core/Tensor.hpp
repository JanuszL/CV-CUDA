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

#ifndef NVCV_PRIV_TENSOR_HPP
#define NVCV_PRIV_TENSOR_HPP

#include "ITensor.hpp"

#include <cuda_runtime.h>

namespace nv::cv::priv {

class Tensor final : public CoreObjectBase<ITensor>
{
public:
    explicit Tensor(NVCVTensorRequirements reqs, IAllocator &alloc);
    ~Tensor();

    static NVCVTensorRequirements CalcRequirements(int32_t numImages, Size2D imgSize, ImageFormat fmt);
    static NVCVTensorRequirements CalcRequirements(int ndim, const int64_t *shape, const PixelType &dtype,
                                                   NVCVTensorLayout layout);

    int32_t        ndim() const override;
    const int64_t *shape() const override;

    const NVCVTensorLayout &layout() const override;

    PixelType dtype() const override;

    IAllocator &alloc() const override;

    void exportData(NVCVTensorData &data) const override;

private:
    IAllocator            &m_alloc;
    NVCVTensorRequirements m_reqs;

    void *m_buffer;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TENSOR_HPP
