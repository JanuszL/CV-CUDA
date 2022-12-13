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

#ifndef NVCV_TENSOR_HPP
#define NVCV_TENSOR_HPP

#include "IImage.hpp"
#include "ITensor.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"
#include "TensorData.hpp"
#include "alloc/IAllocator.hpp"

namespace nv { namespace cv {

// Tensor tensor definition -------------------------------------
class Tensor : public ITensor
{
public:
    using Requirements = NVCVTensorRequirements;
    static Requirements CalcRequirements(const TensorShape &shape, PixelType dtype);
    static Requirements CalcRequirements(int numImages, Size2D imgSize, ImageFormat fmt);

    explicit Tensor(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit Tensor(const TensorShape &shape, PixelType dtype, IAllocator *alloc = nullptr);
    explicit Tensor(int numImages, Size2D imgSize, ImageFormat fmt, IAllocator *alloc = nullptr);
    ~Tensor();

    Tensor(const Tensor &) = delete;

private:
    NVCVTensorHandle doGetHandle() const final;

    NVCVTensorHandle m_handle;
};

// TensorWrapData definition -------------------------------------
using TensorDataCleanupFunc = void(const ITensorData &);

class TensorWrapData : public ITensor
{
public:
    explicit TensorWrapData(const ITensorData &data, std::function<TensorDataCleanupFunc> cleanup = nullptr);
    ~TensorWrapData();

    TensorWrapData(const TensorWrapData &) = delete;

private:
    NVCVTensorHandle doGetHandle() const final;

    static void doCleanup(void *ctx, const NVCVTensorData *data);

    NVCVTensorHandle m_handle;

    std::function<TensorDataCleanupFunc> m_cleanup;
};

// TensorWrapImage definition -------------------------------------
class TensorWrapImage : public ITensor
{
public:
    explicit TensorWrapImage(const IImage &mg);
    ~TensorWrapImage();

    TensorWrapImage(const TensorWrapImage &) = delete;

private:
    NVCVTensorHandle doGetHandle() const final;

    NVCVTensorHandle m_handle;
};

// TensorWrapHandle definition -------------------------------------
// Refers to an external NVCVTensor handle. It doesn't own it.
class TensorWrapHandle : public ITensor
{
public:
    explicit TensorWrapHandle(NVCVTensorHandle handle);

    TensorWrapHandle(const TensorWrapHandle &that);

private:
    NVCVTensorHandle doGetHandle() const final;

    NVCVTensorHandle m_handle;
};

}} // namespace nv::cv

#include "detail/TensorImpl.hpp"

#endif // NVCV_TENSOR_HPP
