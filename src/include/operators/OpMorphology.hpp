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

/**
 * @file OpMorphology.hpp
 *
 * @brief Defines the public C++ Class for the Morphology operation.
 * @defgroup NVCV_CPP_ALGORITHM_MORPHOLOGY Morphology
 * @{
 */

#ifndef NVCV_OP_MORPHOLOGY_HPP
#define NVCV_OP_MORPHOLOGY_HPP

#include "IOperator.hpp"
#include "OpMorphology.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Morphology final : public IOperator
{
public:
    explicit Morphology();

    ~Morphology();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, NVCVMorphologyType morph_type,
                    const cv::Size2D &maskSize, const int2 &anchor, int iteration, const NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Morphology::Morphology()
{
    cv::detail::CheckThrow(nvcvopMorphologyCreate(&m_handle));
    assert(m_handle);
}

inline Morphology::~Morphology()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Morphology::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, NVCVMorphologyType morphType,
                                   const cv::Size2D &maskSize, const int2 &anchor, int iteration,
                                   const NVCVBorderType borderMode)
{
    cv::detail::CheckThrow(nvcvopMorphologySubmit(m_handle, stream, in.handle(), out.handle(), morphType, maskSize.w,
                                                  maskSize.h, anchor.x, anchor.y, iteration, borderMode));
}

inline NVCVOperatorHandle Morphology::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_MORPHOLOGY_HPP
