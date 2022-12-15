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

#include "Tensor.hpp"

#include "Assert.hpp"
#include "CheckError.hpp"
#include "CudaBuffer.hpp"
#include "DataType.hpp"
#include "Hash.hpp"
#include "Image.hpp"
#include "ImageFormat.hpp"
#include "PyUtil.hpp"
#include "String.hpp"

#include <nvcv/TensorShapeInfo.hpp>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace nv::cv {

static size_t ComputeHash(const cv::TensorShape &shape)
{
    using cvpy::ComputeHash;
    return ComputeHash(shape.shape(), shape.layout());
}

} // namespace nv::cv

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

std::shared_ptr<Tensor> Tensor::Create(Shape shape, cv::DataType dtype, std::optional<cv::TensorLayout> layout)
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
        auto tensor = std::static_pointer_cast<Tensor>(vcont[0]);
        NVCV_ASSERT(tensor->dtype() == reqs.dtype);
        return tensor;
    }
}

namespace {

NVCVTensorData FillNVCVTensorData(const py::buffer_info &info, std::optional<cv::TensorLayout> layout,
                                  NVCVTensorBufferType bufType)
{
    NVCVTensorData tensorData = {};

    // dtype ------------
    tensorData.dtype = py::cast<cv::DataType>(ToDType(info));

    // layout ------------
    if (layout)
    {
        tensorData.layout = *layout;
    }

    // ndim ------------
    {
        int ndim = info.ndim == 0 ? 1 : info.ndim;
        if (ndim < 1 || ndim > NVCV_TENSOR_MAX_NDIM)
        {
            throw std::invalid_argument(
                FormatString("Number of dimensions must be between 1 and %d, not %d", NVCV_TENSOR_MAX_NDIM, ndim));
        }
        tensorData.ndim = ndim;
    }

    // shape ------------
    if (info.ndim == 0)
    {
        // according to https://docs.python.org/3/c-api/buffer.html,
        // when ndim is zero, buf points to a scalar, so its shape is [1]
        // info.shape and info.strides are NULL.
        tensorData.shape[0] = 1;
    }
    else
    {
        for (int d = 0; d < info.ndim; ++d)
        {
            tensorData.shape[d] = info.shape[d];
        }
    }

    // buffer type ------------
    tensorData.bufferType = bufType;
    NVCV_ASSERT(bufType == NVCV_TENSOR_BUFFER_STRIDED_CUDA && "Only pitch-linear device buffer supported for now");

    NVCVTensorBufferStrided &dataStrided = tensorData.buffer.strided;

    // pitch ------------
    if (info.ndim == 0)
    {
        // tensor only holds one scalar, to strides is itemsize
        dataStrided.strides[0] = info.itemsize;
    }
    else
    {
        for (int d = 0; d < info.ndim; ++d)
        {
            dataStrided.strides[d] = info.strides[d];
        }
    }

    // Memory buffer ------------
    dataStrided.basePtr = reinterpret_cast<NVCVByte *>(info.ptr);

    return tensorData;
}

NVCVTensorData FillNVCVTensorDataCUDA(const py::buffer_info &info, std::optional<cv::TensorLayout> layout)
{
    return FillNVCVTensorData(info, std::move(layout), NVCV_TENSOR_BUFFER_STRIDED_CUDA);
}

} // namespace

std::shared_ptr<Tensor> Tensor::Wrap(CudaBuffer &buffer, std::optional<cv::TensorLayout> layout)
{
    py::buffer_info info = buffer.request(true);

    cv::TensorDataStridedCuda data{FillNVCVTensorDataCUDA(info, std::move(layout))};

    // This is the key of a tensor wrapper.
    // All tensor wrappers have the same key.
    Tensor::Key key;
    // We take this opportunity to remove from cache all wrappers that aren't
    // being used. They aren't reusable anyway.
    Cache::Instance().removeAllNotInUseMatching(key);

    auto tensor = std::shared_ptr<Tensor>(new Tensor(data, py::cast(buffer.shared_from_this())));

    // Need to add wrappers to cache so that they don't get destroyed by
    // the cuda stream when they're last used, and python script isn't
    // holding a reference to them. If we don't do it, things might break.
    Cache::Instance().add(*tensor);
    return tensor;
}

std::shared_ptr<Tensor> Tensor::WrapImage(Image &img)
{
    Tensor::Key key;
    Cache::Instance().removeAllNotInUseMatching(key);

    auto tensor = std::shared_ptr<Tensor>(new Tensor(img));

    Cache::Instance().add(*tensor);
    return tensor;
}

Tensor::Tensor(const cv::Tensor::Requirements &reqs)
    : m_impl{std::make_unique<cv::Tensor>(reqs)}
    , m_key{reqs}
{
}

Tensor::Tensor(const cv::ITensorData &data, py::object wrappedObject)
    : m_impl{std::make_unique<cv::TensorWrapData>(data)}
    , m_key{}
    , m_wrappedObject(wrappedObject)
{
}

Tensor::Tensor(Image &img)
    : m_impl{std::make_unique<cv::TensorWrapImage>(img.impl())}
    , m_key{}
    , m_wrappedObject(py::cast(img))
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

cv::ITensor &Tensor::impl()
{
    return *m_impl;
}

const cv::ITensor &Tensor::impl() const
{
    return *m_impl;
}

Shape Tensor::shape() const
{
    cv::Shape ishape = m_impl->shape().shape();

    return Shape(ishape.begin(), ishape.end());
}

std::optional<cv::TensorLayout> Tensor::layout() const
{
    const cv::TensorLayout &layout = m_impl->layout();
    if (layout != cv::TensorLayout::NONE)
    {
        return layout;
    }
    else
    {
        return std::nullopt;
    }
}

cv::DataType Tensor::dtype() const
{
    return m_impl->dtype();
}

int Tensor::ndim() const
{
    return m_impl->ndim();
}

Tensor::Key::Key(const cv::Tensor::Requirements &reqs)
    : Key(cv::TensorShape(reqs.shape, reqs.ndim, reqs.layout), static_cast<cv::DataType>(reqs.dtype))
{
}

Tensor::Key::Key(const cv::TensorShape &shape, cv::DataType dtype)
    : m_shape(std::move(shape))
    , m_dtype(dtype)
    , m_wrapper(false)
{
}

size_t Tensor::Key::doGetHash() const
{
    if (m_wrapper)
    {
        return 0; // all wrappers are equal wrt. the cache
    }
    else
    {
        return ComputeHash(m_shape, m_dtype);
    }
}

bool Tensor::Key::doIsEqual(const IKey &that_) const
{
    const Key &that = static_cast<const Key &>(that_);

    // Wrapper key's all compare equal, are they can't be used
    // and whenever we query the cache for wrappers, we really
    // want to get them all (as long as they aren't being used).
    if (m_wrapper && that.m_wrapper)
    {
        return true;
    }
    else if (m_wrapper || that.m_wrapper) // xor
    {
        return false;
    }
    else
    {
        return std::tie(m_shape, m_dtype) == std::tie(that.m_shape, that.m_dtype);
    }
}

auto Tensor::key() const -> const Key &
{
    return m_key;
}

static py::buffer_info ToPyBufferInfo(const cv::ITensorDataStrided &tensorData)
{
    std::vector<ssize_t> shape(tensorData.shape().shape().begin(), tensorData.shape().shape().end());
    std::vector<ssize_t> strides(tensorData.cdata().buffer.strided.strides,
                                 tensorData.cdata().buffer.strided.strides + tensorData.ndim());

    py::dtype dt = py::cast<py::dtype>(py::cast(tensorData.dtype()));

    // There's no direct way to construct a py::buffer_info from data together with a py::dtype.
    // To do that, we first construct a py::array (it accepts py::dtype), and use ".request()"
    // to retrieve the corresponding py::buffer_info.
    // To avoid spurious data copies in py::array ctor, we create this dummy owner.
    py::tuple tmpOwner = py::make_tuple();
    py::array tmp(dt, shape, strides, tensorData.basePtr(), tmpOwner);

    return tmp.request();
}

static py::object ToPython(const cv::ITensorData &imgData, py::object owner)
{
    py::object out;

    auto *stridedData = dynamic_cast<const cv::ITensorDataStrided *>(&imgData);
    if (!stridedData)
    {
        throw std::runtime_error("Only tensors with pitch-linear data can be exported");
    }

    py::buffer_info info = ToPyBufferInfo(*stridedData);
    if (dynamic_cast<const cv::ITensorDataStridedCuda *>(stridedData))
    {
        if (owner)
        {
            return py::cast(std::make_shared<CudaBuffer>(info, false), py::return_value_policy::reference_internal,
                            owner);
        }
        else
        {
            return py::cast(std::make_shared<CudaBuffer>(info, true), py::return_value_policy::take_ownership);
        }
    }
    else
    {
        throw std::runtime_error("Buffer type not supported");
    }
}

py::object Tensor::cuda() const
{
    // Do we need to redefine the cuda object?
    if (!m_cacheCudaObject)
    {
        const cv::ITensorData *tensorData = m_impl->exportData();
        if (!tensorData)
        {
            throw std::runtime_error("Tensor data can't be exported");
        }

        m_cacheCudaObject = ToPython(*tensorData, py::cast(*this));
    }

    return m_cacheCudaObject;
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

    auto p = s.rfind('_');
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
#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) .def_readonly_static(#LAYOUT, &cv::TensorLayout::LAYOUT)
#include <nvcv/TensorLayoutDef.inc>
#undef NVCV_DETAIL_DEF_TLAYOUT
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", &TensorLayoutToString);

    py::implicitly_convertible<py::str, cv::TensorLayout>();

    py::class_<Tensor, std::shared_ptr<Tensor>, Container>(m, "Tensor")
        .def(py::init(&Tensor::CreateForImageBatch), "nimages"_a, "imgsize"_a, "format"_a)
        .def(py::init(&Tensor::Create), "shape"_a, "dtype"_a, "layout"_a = std::nullopt)
        .def_property_readonly("layout", &Tensor::layout)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def("cuda", &Tensor::cuda)
        .def("__repr__", &ToString<Tensor>);

    m.def("as_tensor", &Tensor::Wrap, "buffer"_a, "layout"_a = std::nullopt);
    m.def("as_tensor", &Tensor::WrapImage, "image"_a);
}

} // namespace nv::cvpy
