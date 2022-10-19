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

#ifndef NVCV_PYTHON_TENSOR_HPP
#define NVCV_PYTHON_TENSOR_HPP

#include "Container.hpp"

#include <nvcv/Tensor.hpp>

#include <vector>

namespace nv::cvpy {
namespace py = pybind11;

using Shape  = std::vector<int64_t>;
using Size2D = std::tuple<int, int>;
class CudaBuffer;

Shape CreateShape(const cv::TensorShape &tshape);

class Tensor : public Container
{
public:
    static void Export(py::module &m);

    static std::shared_ptr<Tensor> CreateForImageBatch(int numImages, const Size2D &size, cv::ImageFormat fmt);
    static std::shared_ptr<Tensor> Create(Shape shape, cv::PixelType dtype, std::optional<cv::TensorLayout> layout);

    static std::shared_ptr<Tensor> CreateFromReqs(const cv::Tensor::Requirements &reqs);

    static std::shared_ptr<Tensor> Wrap(CudaBuffer &buffer, std::optional<cv::TensorLayout> layout);

    std::shared_ptr<Tensor>       shared_from_this();
    std::shared_ptr<const Tensor> shared_from_this() const;

    std::optional<cv::TensorLayout> layout() const;
    Shape                           shape() const;
    cv::PixelType                   dtype() const;
    int                             ndim() const;

    cv::ITensor       &impl();
    const cv::ITensor &impl() const;

    class Key final : public IKey
    {
    public:
        explicit Key(const cv::Tensor::Requirements &reqs);
        explicit Key(const cv::TensorShape &shape, cv::PixelType dtype);

    private:
        cv::TensorShape m_shape;
        cv::PixelType   m_dtype;

        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override;

private:
    Tensor(const cv::Tensor::Requirements &reqs);
    Tensor(const NVCVTensorData &data, py::object wrappedObject);

    // m_impl must come before m_key
    std::unique_ptr<cv::ITensor> m_impl;
    Key                          m_key;

    py::object m_wrappedObject; // null if not wrapping
};

std::ostream &operator<<(std::ostream &out, const Tensor &tensor);

} // namespace nv::cvpy

#endif // NVCV_PYTHON_TENSOR_HPP
