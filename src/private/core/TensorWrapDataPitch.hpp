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

#ifndef NVCV_PRIV_TENSOR_WRAPDATAPITCH_HPP
#define NVCV_PRIV_TENSOR_WRAPDATAPITCH_HPP

#include "ITensor.hpp"

#include <cuda_runtime.h>

namespace nv::cv::priv {

class TensorWrapDataPitch final : public ITensor
{
public:
    explicit TensorWrapDataPitch(const NVCVTensorData &tdata, NVCVTensorDataCleanupFunc cleanup, void *ctxCleanup);
    ~TensorWrapDataPitch();

    int32_t        ndim() const override;
    const int64_t *shape() const override;

    const NVCVTensorLayout &layout() const override;

    PixelType dtype() const override;

    IAllocator &alloc() const override;

    void exportData(NVCVTensorData &tdata) const override;

private:
    NVCVTensorData m_tdata;

    NVCVTensorDataCleanupFunc m_cleanup;
    void                     *m_ctxCleanup;

    Version doGetVersion() const override;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TENSOR_WRAPDATAPITCH_HPP