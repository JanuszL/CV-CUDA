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

#include "PixelType.hpp"

#include "Assert.hpp"
#include "String.hpp"

#include <nvcv/PixelType.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <complex>
#include <sstream>

// PixelType is implicitly convertible fromto numpy types such as
// numpy.int8, numpy.complex64, numpy.dtype, etc.

namespace nv::cv {
size_t ComputeHash(const cv::PixelType &pix)
{
    return std::hash<uint64_t>()(static_cast<uint64_t>(pix));
}

} // namespace nv::cv

namespace nv::cvpy {

namespace {

template<class T>
struct IsComplex : std::false_type
{
};

template<class T>
struct IsComplex<std::complex<T>> : std::true_type
{
};

template<class T>
bool FindPixelType(const py::dtype &dt, cv::PixelType *pix)
{
    int       nchannels = 1;
    py::dtype dtbase    = dt;
    if (hasattr(dt, "subdtype"))
    {
        py::object obj = dt.attr("subdtype");
        if (!obj.equal(py::none()))
        {
            auto subdt = py::cast<py::tuple>(obj);
            if (subdt.size() != 2)
            {
                // Malformed? subdtype tuple must have 2 elements.
                return false;
            }
            dtbase = subdt[0];

            // only 1d shape for now
            auto shape = py::cast<py::tuple>(subdt[1]);
            if (shape.size() >= 2)
            {
                return false;
            }

            nchannels = shape.empty() ? 1 : py::cast<int>(shape[0]);
        }
    }

    int itemsize = dtbase.itemsize();

    if (dtbase.equal(py::dtype::of<T>()))
    {
        cv::DataType dataType;
        if (IsComplex<T>::value)
        {
            nchannels = 2;
            itemsize /= 2;
            dataType = cv::DataType::FLOAT;
        }
        else if (std::is_floating_point<T>::value)
        {
            dataType = cv::DataType::FLOAT;
        }
        else if (std::is_signed<T>::value)
        {
            dataType = cv::DataType::SIGNED;
        }
        else if (std::is_unsigned<T>::value)
        {
            dataType = cv::DataType::UNSIGNED;
        }
        else
        {
            NVCV_ASSERT(!"Invalid type");
        }

        // Infer the packing
        cv::PackingParams pp = {};

        pp.byteOrder = cv::ByteOrder::MSB;

        switch (nchannels)
        {
        case 1:
            pp.swizzle = cv::Swizzle::S_X000;
            break;
        case 2:
            pp.swizzle = cv::Swizzle::S_XY00;
            break;
        case 3:
            pp.swizzle = cv::Swizzle::S_XYZ0;
            break;
        case 4:
            pp.swizzle = cv::Swizzle::S_XYZW;
            break;
        default:
            NVCV_ASSERT(!"Invalid number of channels");
        }
        for (int i = 0; i < nchannels; ++i)
        {
            pp.bits[i] = static_cast<int>(itemsize * 8);
        }
        cv::Packing packing = MakePacking(pp);

        // Finally, infer the pixel type
        NVCV_ASSERT(pix != nullptr);
        *pix = cv::PixelType{dataType, packing};
        return true;
    }
    else
    {
        return false;
    }
}

// clang-format off
using SupportedBaseTypes = std::tuple<
      std::complex<float>, // must come before float
      std::complex<double>, // must come before double
      float, double,
      uint8_t, int8_t,
      uint16_t, int16_t,
      uint32_t, int32_t,
      uint64_t, int64_t
>;

// clang-format on

template<class... TT>
std::optional<cv::PixelType> SelectPixelType(std::tuple<TT...>, const py::dtype &dt)
{
    cv::PixelType pixtype;

    if ((FindPixelType<TT>(dt, &pixtype) || ...))
    {
        return pixtype;
    }
    else
    {
        return std::nullopt;
    }
}

std::optional<cv::PixelType> ToPixelType(const py::dtype &dt)
{
    return SelectPixelType(SupportedBaseTypes(), dt);
}

template<class T>
bool FindDType(T *, const cv::PixelType &pix, py::dtype *dt)
{
    int nchannels = pix.numChannels();
    int itemsize  = pix.bitsPerPixel() / 8;

    if (sizeof(T) != itemsize / nchannels)
    {
        return false;
    }

    cv::DataType dataType = pix.dataType();

    if ((std::is_floating_point_v<T> && dataType == cv::DataType::FLOAT)
        || (std::is_integral_v<T> && std::is_signed_v<T> && dataType == cv::DataType::SIGNED)
        || (std::is_integral_v<T> && std::is_unsigned_v<T> && dataType == cv::DataType::UNSIGNED))
    {
        NVCV_ASSERT(dt != nullptr);

        *dt = py::dtype::of<T>();

        // pix has multiple components?
        if (cv::cuda::NumElements<T> != nchannels)
        {
            // Create a dtype with multiple components too, with shape argument
            *dt = py::dtype(FormatString("%d%c", nchannels, dt->char_()));
        }
        return true;
    }
    else
    {
        return false;
    }
}

template<class T>
bool FindDType(std::complex<T> *, const cv::PixelType &pix, py::dtype *dt)
{
    cv::DataType dataType  = pix.dataType();
    int          nchannels = pix.numChannels();
    int          itemsize  = pix.bitsPerPixel() / 8;

    if (dataType == cv::DataType::FLOAT && sizeof(std::complex<T>) == itemsize && nchannels == 2)
    {
        NVCV_ASSERT(dt != nullptr);
        *dt = py::dtype::of<std::complex<T>>();
        return true;
    }
    else
    {
        return false;
    }
}

template<class... TT>
py::dtype SelectDType(std::tuple<TT...>, const cv::PixelType &pix)
{
    py::dtype dt;

    (FindDType((TT *)nullptr, pix, &dt) || ...);

    return dt;
}

py::dtype ToDType(cv::PixelType pix)
{
    return SelectDType(SupportedBaseTypes(), pix);
}

} // namespace

static std::string PixelTypeToString(cv::PixelType type)
{
    const char *str = nvcvPixelTypeGetName(type);

    const char *prefix = "NVCV_PIXEL_TYPE_";

    std::ostringstream out;

    out << "nvcv.";

    if (strncmp(str, prefix, strlen(prefix)) == 0)
    {
        out << "Type." << str + strlen(prefix);
    }
    else
    {
        prefix = "PixelType";
        if (strncmp(str, prefix, strlen(prefix)) == 0)
        {
            out << "Type" << str + strlen(prefix);
        }
        else
        {
            out << "<Unknown type: " << str << '>';
        }
    }

    return out.str();
}

void ExportPixelType(py::module &m)
{
    py::class_<cv::PixelType> type(m, "Type");

#define DEF(F)     type.def_readonly_static(#F, &cv::TYPE_##F);
// for formats that begin with a number, we must prepend it with underscore to make
// it a valid python identifier
#define DEF_NUM(F) type.def_readonly_static("_" #F, &cv::TYPE_##F);

#include "NVCVPythonPixelTypeDefs.inc"

#undef DEF
#undef DEF_NUM

    type.def_property_readonly("components", &cv::PixelType::numChannels);
    type.def(py::init<cv::PixelType>());
    type.def(py::init<>());

    type.def("__repr__", &PixelTypeToString);
    type.def(py::self == py::self);
    type.def(py::self != py::self);
    type.def(py::self < py::self);

    py::implicitly_convertible<py::dtype, cv::PixelType>();
}

} // namespace nv::cvpy

namespace pybind11::detail {

bool type_caster<nv::cv::PixelType>::load(handle src, bool)
{
    const type_info *pixtype = get_type_info(typeid(nv::cv::PixelType));
    if (Py_TYPE(src.ptr()) == pixtype->type)
    {
        value_and_holder vh = reinterpret_cast<instance *>(src.ptr())->get_value_and_holder();
        value               = *vh.template holder<nv::cv::PixelType *>();
        return true;
    }
    else
    {
        PyObject *ptr = nullptr;
        if (detail::npy_api::get().PyArray_DescrConverter_(src.ptr(), &ptr) == 0 || !ptr)
        {
            PyErr_Clear();
            return false;
        }
        dtype dt = dtype::from_args(reinterpret_steal<object>(ptr));

        if (std::optional<nv::cv::PixelType> pix = nv::cvpy::ToPixelType(dt))
        {
            value = *pix;
            return true;
        }
        else
        {
            return false;
        }
    }
}

handle type_caster<nv::cv::PixelType>::cast(nv::cv::PixelType type, return_value_policy /* policy */,
                                            handle /*parent */)
{
    dtype dt = nv::cvpy::ToDType(type);

    // without the increfs, we get 6 of these...
    // *** Reference count error detected: an attempt was made to deallocate the dtype 6 (I) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 3 (h) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 4 (H) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 1 (b) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 2 (B) ***
    // and also a segfault when using the nvcv struct types in some tests.
    // It *really* looks like we have to incref here.

    if (dt)
    {
        Py_INCREF(dt.ptr());
    }

    return dt;
}
} // namespace pybind11::detail
