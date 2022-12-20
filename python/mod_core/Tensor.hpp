/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_PYTHON_PRIV_TENSOR_HPP
#define NVCV_PYTHON_PRIV_TENSOR_HPP

#include "Container.hpp"
#include "Size.hpp"

#include <nvcv/Tensor.hpp>
#include <pybind11/numpy.h>

#include <vector>

namespace nv::cvpy::priv {
namespace py = pybind11;

using Shape = std::vector<int64_t>;

class CudaBuffer;
class Image;

Shape CreateShape(const cv::TensorShape &tshape);

class Tensor : public Container
{
public:
    static void Export(py::module &m);

    static std::shared_ptr<Tensor> CreateForImageBatch(int numImages, const Size2D &size, cv::ImageFormat fmt);
    static std::shared_ptr<Tensor> Create(Shape shape, cv::DataType dtype, std::optional<cv::TensorLayout> layout);

    static std::shared_ptr<Tensor> CreateFromReqs(const cv::Tensor::Requirements &reqs);

    static std::shared_ptr<Tensor> Wrap(CudaBuffer &buffer, std::optional<cv::TensorLayout> layout);
    static std::shared_ptr<Tensor> WrapImage(Image &img);

    std::shared_ptr<Tensor>       shared_from_this();
    std::shared_ptr<const Tensor> shared_from_this() const;

    std::optional<cv::TensorLayout> layout() const;
    Shape                           shape() const;
    cv::DataType                    dtype() const;
    int                             rank() const;

    cv::ITensor       &impl();
    const cv::ITensor &impl() const;

    class Key final : public IKey
    {
    public:
        explicit Key()
            : m_wrapper(true)
        {
        }

        explicit Key(const cv::Tensor::Requirements &reqs);
        explicit Key(const cv::TensorShape &shape, cv::DataType dtype);

    private:
        cv::TensorShape m_shape;
        cv::DataType    m_dtype;
        bool            m_wrapper;

        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override;

    py::object cuda() const;

private:
    Tensor(const cv::Tensor::Requirements &reqs);
    Tensor(const cv::ITensorData &data, py::object wrappedObject);
    Tensor(Image &img);

    // m_impl must come before m_key
    std::unique_ptr<cv::ITensor> m_impl;
    Key                          m_key;

    mutable py::object                      m_cacheCudaObject;
    mutable std::optional<cv::TensorLayout> m_cacheCudaObjectLayout;

    py::object m_wrappedObject; // null if not wrapping
};

std::ostream &operator<<(std::ostream &out, const Tensor &tensor);

} // namespace nv::cvpy::priv

#endif // NVCV_PYTHON_PRIV_TENSOR_HPP
