/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ExternalBuffer.hpp"

#include "DataType.hpp"

#include <common/Assert.hpp>
#include <common/PyUtil.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <functional> // for std::multiplies

namespace nvcvpy::priv {

using namespace py::literals;

static void CheckValidCUDABuffer(const void *ptr)
{
    if (ptr == nullptr)
    {
        throw std::runtime_error("NULL CUDA buffer not accepted");
    }

    cudaPointerAttributes attrs = {};
    cudaError_t           err   = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset the cuda error (if any)
    if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered)
    {
        throw std::runtime_error("Buffer is not CUDA-accessible");
    }
}

static DLPackTensor CopyBuffer(const DLTensor &src)
{
    struct StrideIndex
    {
        int64_t stride;
        int     idx;
    };

    std::vector<StrideIndex> stridesIndex(src.ndim);

    if (src.strides == nullptr)
    {
        // packed, row-major.
        stridesIndex.back().stride = 1;
        stridesIndex.back().idx    = src.ndim - 1;
        for (int i = src.ndim - 1; i > 0; --i)
        {
            stridesIndex[i - 1].stride = stridesIndex[i].stride * src.shape[i];
            stridesIndex[i - 1].idx    = i - 1;
        }
    }
    else
    {
        for (int i = 0; i < src.ndim; ++i)
        {
            if (src.strides[i] <= 0)
            {
                throw std::runtime_error("Dimension stride must be > 0");
            }

            stridesIndex[i].stride = src.strides[i];
            stridesIndex[i].idx    = i;
        }

        std::sort(stridesIndex.begin(), stridesIndex.end(),
                  [](const StrideIndex &a, const StrideIndex &b) { return a.stride >= b.stride; });
    }

    int elemStrideBytes = (src.dtype.lanes * src.dtype.bits + 7) / 8;

    int numCols        = src.shape[stridesIndex.back().idx];
    int numRows        = std::reduce(src.shape, src.shape + src.ndim, 1, std::multiplies<>()) / numCols;
    int rowStrideBytes = numCols * elemStrideBytes;

    void       *newData;
    size_t      newPitch;
    cudaError_t err = cudaMallocPitch(&newData, &newPitch, rowStrideBytes, numRows);
    if (err != cudaSuccess)
    {
        std::ostringstream ss;
        ss << "Error allocating " << rowStrideBytes * numRows << " bytes of cuda memory: " << cudaGetErrorName(err)
           << " - " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }

    DLPackTensor dlTensor;
    {
        DLManagedTensor dlManagedTensor = {};
        dlManagedTensor.deleter         = [](DLManagedTensor *self)
        {
            cudaFree(self->dl_tensor.data);
            delete[] self->dl_tensor.strides;
            delete[] self->dl_tensor.shape;
        };

        dlTensor = DLPackTensor{std::move(dlManagedTensor)};
    }

    dlTensor->data               = newData;
    dlTensor->device.device_type = kDLCUDA;
    // TODO: set the correct device id
    dlTensor->device.device_id = 0;

    dlTensor->ndim = src.ndim;

    // for now dlTensor will be permuted so that its strides are in
    // decreasing order.

    std::vector<int64_t> shape(src.ndim);
    for (int i = 0; i < src.ndim; ++i)
    {
        shape[i] = src.shape[stridesIndex[i].idx];
    }

    std::vector<int64_t> dstStrides(src.ndim);
    dstStrides[src.ndim - 1] = elemStrideBytes;
    if (src.ndim >= 2)
    {
        dstStrides[src.ndim - 2] = newPitch / elemStrideBytes;
        for (int i = src.ndim - 2; i > 0; --i)
        {
            dstStrides[i - 1] = dstStrides[i] * shape[i];
        }
    }

    int numRowsCopy = src.ndim > 1 ? dlTensor->shape[src.ndim - 2] : 1;

    int idxFirstPackedRows = std::max<int>(0, src.ndim - 2);
    for (int i = src.ndim - 3; i >= 0; --i)
    {
        if (dlTensor->strides[i] == dlTensor->strides[i + 1] * dlTensor->shape[i + 1])
        {
            numRowsCopy *= dlTensor->shape[i];
            idxFirstPackedRows = i;
        }
    }

    std::function<void(int, const std::byte *, std::byte *)> copy;
    copy = [&copy, &numRowsCopy, &idxFirstPackedRows, &dlTensor, &dstStrides, &rowStrideBytes, &elemStrideBytes,
            &stridesIndex](int idx, const std::byte *srcData, std::byte *dstData)
    {
        if (idx == idxFirstPackedRows)
        {
            cudaError_t err = cudaMemcpy2D(dstData, dstStrides[idx] * elemStrideBytes, srcData,
                                           stridesIndex[idx].stride * elemStrideBytes, rowStrideBytes, numRowsCopy,
                                           cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                std::ostringstream ss;
                ss << "Error copying cuda buffer: " << cudaGetErrorName(err) << " - " << cudaGetErrorString(err);
                throw std::runtime_error(ss.str());
            }
        }
        else
        {
            for (int i = 0; i < idx; ++i)
            {
                copy(idx + 1, srcData, dstData);
                srcData += stridesIndex[idx].stride * elemStrideBytes;
                dstData += dstStrides[idx] * elemStrideBytes;
            }
        }
    };

    copy(0, static_cast<const std::byte *>(src.data), static_cast<std::byte *>(newData));

    // now permute the tensor to the original order

    dlTensor->byte_offset = 0;
    dlTensor->dtype       = src.dtype;

    dlTensor->shape = new int64_t[src.ndim];
    std::copy_n(dlTensor->shape, dlTensor->ndim, src.shape);

    dlTensor->strides = new int64_t[src.ndim];
    for (int i = 0; i < src.ndim; ++i)
    {
        dlTensor->strides[stridesIndex[i].idx] = dstStrides[i];
    }

    return dlTensor;
}

static std::string ToFormatString(const DLDataType &dtype)
{
    // TODO: these must be a more efficient way to retrieve the
    // format string from a dtype...
    py::array tmp(ToDType(ToNVCVDataType(dtype)), py::array::ShapeContainer{});
    return tmp.request().format;
}

std::shared_ptr<ExternalBuffer> ExternalBuffer::Create(DLPackTensor &&dlPackTensor, bool copy, py::object wrappedObj)
{
    std::shared_ptr<ExternalBuffer> buf(new ExternalBuffer(std::move(dlPackTensor), copy, std::move(wrappedObj)));
    return buf;
}

ExternalBuffer::ExternalBuffer(DLPackTensor &&dlTensor, bool copy, py::object wrappedObj)
    : m_wrappedObj(wrappedObj)
{
    if (!IsCudaAccessible(dlTensor->device.device_type))
    {
        throw std::runtime_error("Only CUDA memory buffers can be wrapped");
    }

    if (dlTensor->data != nullptr)
    {
        CheckValidCUDABuffer(dlTensor->data);
    }

    if (copy && dlTensor->data != nullptr)
    {
        m_dlTensor = CopyBuffer(*dlTensor);
    }
    else
    {
        m_dlTensor = std::move(dlTensor);
    }
}

py::object ExternalBuffer::shape() const
{
    std::vector<ssize_t> shape(m_dlTensor->ndim);
    std::copy_n(m_dlTensor->shape, m_dlTensor->ndim, shape.begin());

    return py::cast(std::move(shape));
}

py::object ExternalBuffer::dtype() const
{
    return ToDType(ToNVCVDataType(m_dlTensor->dtype));
}

void *ExternalBuffer::data() const
{
    return m_dlTensor->data;
}

bool ExternalBuffer::load(PyObject *o)
{
    if (!o)
    {
        return false;
    }

    py::object tmp = py::reinterpret_borrow<py::object>(o);

    if (hasattr(tmp, "__cuda_array_interface__"))
    {
        py::dict iface = tmp.attr("__cuda_array_interface__").cast<py::dict>();

        if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data")
            || !iface.contains("version"))
        {
            return false;
        }

        int version = iface["version"].cast<int>();
        if (version < 2)
        {
            return false;
        }

        DLPackTensor dlTensor;
        {
            DLManagedTensor dlManagedTensor = {};
            dlManagedTensor.deleter         = [](DLManagedTensor *self)
            {
                delete[] self->dl_tensor.shape;
                delete[] self->dl_tensor.strides;
            };
            dlTensor = DLPackTensor{std::move(dlManagedTensor)};
        }

        dlTensor->byte_offset = 0;

        // TODO: infer the device type from the memory buffer
        dlTensor->device.device_type = kDLCUDA;
        // TODO: infer the device from the memory buffer
        dlTensor->device.device_id = 0;

        // Convert data
        py::tuple tdata = iface["data"].cast<py::tuple>();
        void     *ptr   = reinterpret_cast<void *>(tdata[0].cast<long>());
        CheckValidCUDABuffer(ptr);
        dlTensor->data = ptr;

        // Convert DataType
        py::dtype dt = util::ToDType(iface["typestr"].cast<std::string>());
        if (std::optional<nvcv::DataType> dtype = ToNVCVDataType(dt))
        {
            dlTensor->dtype = ToDLDataType(*dtype);
        }

        // Convert ndim
        py::tuple shape = iface["shape"].cast<py::tuple>();
        dlTensor->ndim  = shape.size();

        // Convert shape
        dlTensor->shape = new int64_t[dlTensor->ndim];
        for (int i = 0; i < dlTensor->ndim; ++i)
        {
            dlTensor->shape[i] = shape[i].cast<long>();
        }

        // Convert strides
        dlTensor->strides = new int64_t[dlTensor->ndim];
        if (iface.contains("strides") && !iface["strides"].is_none())
        {
            py::tuple strides = iface["strides"].cast<py::tuple>();
            for (int i = 0; i < dlTensor->ndim; ++i)
            {
                dlTensor->strides[i] = strides[i].cast<long>();
                if (dlTensor->strides[i] % dt.itemsize() != 0)
                {
                    throw std::runtime_error("Stride must be a multiple of the element size in bytes");
                }
                dlTensor->strides[i] /= dt.itemsize();
            }
        }
        else
        {
            // If strides isn't defined, according to cuda array interface, we must
            // set them up for packed, row-major strides.
            dlTensor->strides[dlTensor->ndim - 1] = 1;
            for (int i = dlTensor->ndim - 1; i > 0; --i)
            {
                dlTensor->strides[i - 1] = dlTensor->strides[i] * dlTensor->shape[i];
            }
        }

        if (dlTensor->ndim >= 1)
        {
            m_wrappedObj              = tmp; // hold the reference to the wrapped object
            m_cacheCudaArrayInterface = std::move(iface);
            m_dlTensor                = std::move(dlTensor);
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}

std::optional<py::dict> ExternalBuffer::cudaArrayInterface() const
{
    if (!m_cacheCudaArrayInterface)
    {
        if (!IsCudaAccessible(m_dlTensor->device.device_type))
        {
            return std::nullopt;
        }

        nvcv::DataType dataType = ToNVCVDataType(m_dlTensor->dtype);

        NVCV_ASSERT(dataType.strideBytes() * 8 == m_dlTensor->dtype.bits);
        NVCV_ASSERT(m_dlTensor->dtype.bits % 8 == 0);
        int elemStrideBytes = m_dlTensor->dtype.bits / 8;

        py::object strides;

        if (m_dlTensor->strides == nullptr)
        {
            strides = py::none();
        }
        else
        {
            std::vector<ssize_t> vStrides(m_dlTensor->ndim);
            for (size_t i = 0; i < vStrides.size(); ++i)
            {
                vStrides[i] = m_dlTensor->strides[i] * elemStrideBytes;
            }
            strides = py::cast(vStrides);
        }

        std::string format = ToFormatString(m_dlTensor->dtype);

        // clang-format off
        m_cacheCudaArrayInterface = py::dict
        {
            "shape"_a = this->shape(),
            "strides"_a = strides,
            "typestr"_a = format,
            "data"_a = py::make_tuple(reinterpret_cast<long>(m_dlTensor->data), false /* read/write */),
            "version"_a = 2
        };
    }

    return *m_cacheCudaArrayInterface;
}

const DLTensor &ExternalBuffer::dlTensor() const
{
    return *m_dlTensor;
}

void ExternalBuffer::Export(py::module &m)
{
    py::class_<ExternalBuffer, std::shared_ptr<ExternalBuffer>>(m, "ExternalBuffer", py::dynamic_attr())
        .def_property_readonly("shape", &ExternalBuffer::shape)
        .def_property_readonly("dtype", &ExternalBuffer::dtype)
        .def("__getattr__", [](std::shared_ptr<ExternalBuffer> buf, std::string name) -> py::object
             {
                if(name == "__cuda_array_interface__")
                {
                    // If we expose cuda array interface,
                    if(auto iface = buf->cudaArrayInterface())
                    {
                        // return it
                        return *iface;
                    }
                }
                throw std::runtime_error("Object "+py::str(py::cast(buf)).cast<std::string>() +" doesn't have attribute "+name);
             });
}

} // namespace nv::vpi::python

namespace pybind11::detail {

namespace priv = nvcvpy::priv;

// Python -> C++
bool type_caster<priv::ExternalBuffer>::load(handle src, bool implicit_conv)
{
    PyTypeObject *srctype = Py_TYPE(src.ptr());
    const type_info *cuda_buffer_type = get_type_info(typeid(priv::ExternalBuffer));

    // src's type is ExternalBuffer?
    if(srctype == cuda_buffer_type->type)
    {
        // We know it's managed by a shared pointer (holder), let's use it
        value_and_holder vh = reinterpret_cast<instance *>(src.ptr())->get_value_and_holder();
        value = vh.template holder<std::shared_ptr<priv::ExternalBuffer>>();
        NVCV_ASSERT(value != nullptr);
        src.inc_ref();
        return true;
    }
    // If not, it could be an object that implements that __cuda_array_interface, let's try to
    // create a ExternalBuffer out of it.
    else
    {
        value.reset(new priv::ExternalBuffer);
        return value->load(src.ptr());
    }
}

} // namespace pybind11::detail
