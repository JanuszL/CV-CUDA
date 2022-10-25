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

#include "Tensor.hpp"

#include "Assert.hpp"
#include "Hash.hpp"
#include "ImageFormat.hpp"
#include "PixelType.hpp"
#include "PyUtil.hpp"
#include "String.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace nv::cvpy {

Shape CreateShape(const cv::TensorShape &tshape)
{
    return Shape{tshape.shape().begin(), tshape.shape().end()};
}

std::shared_ptr<Tensor> Tensor::CreateForImageBatch(int numImages, const Size2D &size, cv::ImageFormat fmt)
{
    cv::Tensor::Requirements reqs
        = cv::Tensor::CalcRequirements(numImages, cv::Size2D{std::get<0>(size), std::get<1>(size)}, fmt);
    return CreateFromReqs(reqs);
}

std::shared_ptr<Tensor> Tensor::Create(Shape shape, cv::PixelType dtype, std::optional<cv::TensorLayout> layout)
{
    if (!layout)
    {
        layout = cv::TensorLayout::NONE;
    }

    cv::Tensor::Requirements reqs
        = cv::Tensor::CalcRequirements(cv::TensorShape(shape.data(), shape.size(), *layout), dtype);
    return CreateFromReqs(reqs);
}

std::shared_ptr<Tensor> Tensor::CreateFromReqs(const cv::Tensor::Requirements &reqs)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{reqs});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Tensor> tensor(new Tensor(reqs));
        Cache::Instance().add(*tensor);
        return tensor;
    }
    else
    {
        // Get the first one
        return std::static_pointer_cast<Tensor>(vcont[0]);
    }
}

Tensor::Tensor(const cv::Tensor::Requirements &reqs)
    : m_impl(reqs)
    , m_key{reqs}
{
}

std::shared_ptr<Tensor> Tensor::shared_from_this()
{
    return std::static_pointer_cast<Tensor>(Container::shared_from_this());
}

std::shared_ptr<const Tensor> Tensor::shared_from_this() const
{
    return std::static_pointer_cast<const Tensor>(Container::shared_from_this());
}

cv::Tensor &Tensor::impl()
{
    return m_impl;
}

const cv::Tensor &Tensor::impl() const
{
    return m_impl;
}

Shape Tensor::shape() const
{
    cv::Shape ishape = m_impl.shape().shape();

    return Shape(ishape.begin(), ishape.end());
}

std::optional<cv::TensorLayout> Tensor::layout() const
{
    const cv::TensorLayout &layout = m_impl.layout();
    if (layout != cv::TensorLayout::NONE)
    {
        return layout;
    }
    else
    {
        return std::nullopt;
    }
}

cv::PixelType Tensor::dtype() const
{
    return m_impl.dtype();
}

int Tensor::ndim() const
{
    return m_impl.ndim();
}

Tensor::Key::Key(const cv::Tensor::Requirements &reqs)
    : Key(Shape(reqs.shape, reqs.shape + reqs.ndim), static_cast<cv::PixelType>(reqs.dtype),
          static_cast<cv::TensorLayout>(reqs.layout))
{
}

Tensor::Key::Key(Shape shape, cv::PixelType dtype, cv::TensorLayout layout)
    : m_shape(std::move(shape))
    , m_dtype(dtype)
    , m_layout(layout)
{
}

size_t Tensor::Key::doGetHash() const
{
    return ComputeHash(m_shape, m_dtype, m_layout);
}

bool Tensor::Key::doIsEqual(const IKey &that_) const
{
    const Key &that = static_cast<const Key &>(that_);

    return std::tie(m_layout, m_shape, m_dtype) == std::tie(that.m_layout, that.m_shape, that.m_dtype);
}

auto Tensor::key() const -> const Key &
{
    return m_key;
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor)
{
    return out << "<nvcv.Tensor shape=" << tensor.impl().shape()
               << " dtype=" << py::cast(tensor.dtype()).str().cast<std::string>() << '>';
}

static std::string TensorLayoutToString(const cv::TensorLayout &layout)
{
    std::ostringstream ss;
    ss << layout;
    std::string s = ss.str();

    int p = s.rfind('_');
    if (p != s.npos)
    {
        return s.substr(p + 1);
    }
    else
    {
        return s;
    }
}

void Tensor::Export(py::module &m)
{
    using namespace py::literals;

    py::class_<cv::TensorLayout>(m, "TensorLayout")
        .def(py::init<const char *>())
        .def_readonly_static("NCHW", &cv::TensorLayout::NCHW)
        .def_readonly_static("NHWC", &cv::TensorLayout::NHWC)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", &TensorLayoutToString);

    py::class_<Tensor, std::shared_ptr<Tensor>, Container>(m, "Tensor")
        .def(py::init(&Tensor::CreateForImageBatch), "nimages"_a, "imgsize"_a, "format"_a)
        .def(py::init(&Tensor::Create), "shape"_a, "dtype"_a, "layout"_a = std::nullopt)
        .def_property_readonly("layout", &Tensor::layout)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def("__repr__", &ToString<Tensor>);
}

} // namespace nv::cvpy
