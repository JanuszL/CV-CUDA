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

class Tensor final : public ITensor
{
public:
    explicit Tensor(NVCVTensorRequirements reqs, IAllocator &alloc);
    ~Tensor();

    static NVCVTensorRequirements CalcRequirements(int32_t numImages, Size2D imgSize, ImageFormat fmt);
    static NVCVTensorRequirements CalcRequirements(const DimsNCHW &dims, ImageFormat fmt);

    const Shape &shape() const override;

    NVCVTensorLayout layout() const override;
    DimsNCHW         dims() const override;

    ImageFormat format() const override;

    IAllocator &alloc() const override;

    void exportData(NVCVTensorData &data) const override;

private:
    IAllocator            &m_alloc;
    NVCVTensorRequirements m_reqs;

    void *m_buffer;

    Version doGetVersion() const override;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TENSOR_HPP
