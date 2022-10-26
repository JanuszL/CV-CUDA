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
#include "CudaBuffer.hpp"
#include "Hash.hpp"
#include "Image.hpp"
#include "ImageFormat.hpp"
#include "PixelType.hpp"
#include "PyUtil.hpp"
#include "String.hpp"

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

namespace {

NVCVTensorData FillNVCVTensorData(const py::buffer_info &info, std::optional<cv::TensorLayout> layout,
                                  NVCVTensorBufferType bufType)
{
    NVCVTensorData tensorData = {};

    // dtype ------------
    tensorData.dtype = py::cast<cv::PixelType>(ToDType(info));

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
        // info.shape and info.pitchBytes are NULL.
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
    NVCV_ASSERT(bufType == NVCV_TENSOR_BUFFER_PITCH_DEVICE && "Only pitch-linear device buffer supported for now");

    NVCVTensorBufferPitch &dataPitch = tensorData.buffer.pitch;

    // pitch ------------
    if (info.ndim == 0)
    {
        // tensor only holds one scalar, to pitchBytes is itemsize
        dataPitch.pitchBytes[0] = info.itemsize;
    }
    else
    {
        for (int d = 0; d < info.ndim; ++d)
        {
            dataPitch.pitchBytes[d] = info.strides[d];
        }
    }

    // data ------------
    dataPitch.data = info.ptr;

    return tensorData;
}

NVCVTensorData FillNVCVTensorDataCUDA(const py::buffer_info &info, std::optional<cv::TensorLayout> layout)
{
    return FillNVCVTensorData(info, std::move(layout), NVCV_TENSOR_BUFFER_PITCH_DEVICE);
}

} // namespace

std::shared_ptr<Tensor> Tensor::Wrap(CudaBuffer &buffer, std::optional<cv::TensorLayout> layout)
{
    py::buffer_info info = buffer.request(true);

    NVCVTensorData data = FillNVCVTensorDataCUDA(info, std::move(layout));

    return std::shared_ptr<Tensor>(new Tensor(data, py::cast(buffer.shared_from_this())));
}

std::shared_ptr<Tensor> Tensor::WrapImage(Image &img)
{
    return std::shared_ptr<Tensor>(new Tensor(img));
}

Tensor::Tensor(const cv::Tensor::Requirements &reqs)
    : m_impl{std::make_unique<cv::Tensor>(reqs)}
    , m_key{reqs}
{
}

Tensor::Tensor(const NVCVTensorData &data, py::object wrappedObject)
    : m_impl{std::make_unique<cv::TensorWrapData>(cv::TensorDataWrap{data})}
    , m_key{m_impl->shape(), m_impl->dtype()}
    , m_wrappedObject(wrappedObject)
{
}

Tensor::Tensor(Image &img)
    : m_impl{std::make_unique<cv::TensorWrapImage>(img.impl())}
    , m_key{m_impl->shape(), m_impl->dtype()}
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

cv::PixelType Tensor::dtype() const
{
    return m_impl->dtype();
}

int Tensor::ndim() const
{
    return m_impl->ndim();
}

Tensor::Key::Key(const cv::Tensor::Requirements &reqs)
    : Key(cv::TensorShape(reqs.shape, reqs.ndim, reqs.layout), static_cast<cv::PixelType>(reqs.dtype))
{
}

Tensor::Key::Key(const cv::TensorShape &shape, cv::PixelType dtype)
    : m_shape(std::move(shape))
    , m_dtype(dtype)
{
}

size_t Tensor::Key::doGetHash() const
{
    return ComputeHash(m_shape, m_dtype);
}

bool Tensor::Key::doIsEqual(const IKey &that_) const
{
    const Key &that = static_cast<const Key &>(that_);

    return std::tie(m_shape, m_dtype) == std::tie(that.m_shape, that.m_dtype);
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
        .def("__repr__", &ToString<Tensor>);

    m.def("as_tensor", &Tensor::Wrap, "buffer"_a, "layout"_a = std::nullopt);
    m.def("as_tensor", &Tensor::WrapImage, "image"_a);
}

} // namespace nv::cvpy
