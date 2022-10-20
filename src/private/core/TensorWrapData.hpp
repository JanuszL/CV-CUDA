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

#ifndef NVCV_PRIV_TENSOR_WRAPDATA_HPP
#define NVCV_PRIV_TENSOR_WRAPDATA_HPP

#include "ITensor.hpp"

#include <cuda_runtime.h>

namespace nv::cv::priv {

class TensorWrapData final : public ITensor
{
public:
    explicit TensorWrapData(const NVCVTensorData &data, NVCVTensorDataCleanupFunc cleanup, void *ctxCleanup);
    ~TensorWrapData();

    int32_t        ndim() const override;
    const int32_t *shape() const override;

    NVCVTensorLayout layout() const override;
    DimsNCHW         dims() const override;

    PixelType dtype() const override;

    IAllocator &alloc() const override;

    void exportData(NVCVTensorData &data) const override;

private:
    NVCVTensorData m_data;

    NVCVTensorDataCleanupFunc m_cleanup;
    void                     *m_ctxCleanup;

    Version doGetVersion() const override;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TENSOR_WRAPDATA_HPP
