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
 * @file OpErase.hpp
 *
 * @brief Defines the public C++ Class for the erase operation.
 * @defgroup NVCV_CPP_ALGORITHM_ERASE Erase
 * @{
 */

#ifndef NVCV_OP_ERASE_HPP
#define NVCV_OP_ERASE_HPP

#include "IOperator.hpp"
#include "OpErase.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Erase final : public IOperator
{
public:
    explicit Erase(int max_num_erasing_area);

    ~Erase();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::ITensor &anchor_x,
                    cv::ITensor &anchor_y, cv::ITensor &erasing_w, cv::ITensor &erasing_h, cv::ITensor &erasing_c,
                    cv::ITensor &values, cv::ITensor &imgIdx, bool random, unsigned int seed, bool inplace);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Erase::Erase(int max_num_erasing_area)
{
    cv::detail::CheckThrow(nvcvopEraseCreate(&m_handle, max_num_erasing_area));
    assert(m_handle);
}

inline Erase::~Erase()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Erase::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::ITensor &anchor_x,
                              cv::ITensor &anchor_y, cv::ITensor &erasing_w, cv::ITensor &erasing_h,
                              cv::ITensor &erasing_c, cv::ITensor &values, cv::ITensor &imgIdx, bool random,
                              unsigned int seed, bool inplace)
{
    cv::detail::CheckThrow(nvcvopEraseSubmit(
        m_handle, stream, in.handle(), out.handle(), anchor_x.handle(), anchor_y.handle(), erasing_w.handle(),
        erasing_h.handle(), erasing_c.handle(), values.handle(), imgIdx.handle(), random, seed, inplace));
}

inline NVCVOperatorHandle Erase::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_ERASE_HPP
