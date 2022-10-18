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

#ifndef NVCV_PYTHON_CUDA_BUFFER_HPP
#define NVCV_PYTHON_CUDA_BUFFER_HPP

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace py = pybind11;

class CudaBuffer final : public std::enable_shared_from_this<CudaBuffer>
{
public:
    static void Export(py::module &m);

    CudaBuffer(CudaBuffer &&that) = delete;

    explicit CudaBuffer(const py::buffer_info &data, bool copy = false);

    ~CudaBuffer();

    py::dict cuda_interface() const;

    py::buffer_info request(bool writable = false) const;

    py::object shape() const;
    py::object dtype() const;

    void *data() const;

    bool load(PyObject *o);

private:
    friend py::detail::type_caster<CudaBuffer>;
    CudaBuffer();

    py::dict m_cudaArrayInterface;
    bool     m_owns;
};

} // namespace nv::cvpy

namespace PYBIND11_NAMESPACE { namespace detail {

template<>
struct type_caster<nv::cvpy::CudaBuffer> : public type_caster_base<nv::cvpy::CudaBuffer>
{
    using type = nv::cvpy::CudaBuffer;
    using Base = type_caster_base<type>;

public:
    PYBIND11_TYPE_CASTER(std::shared_ptr<type>, const_name("nvcv.cuda.Buffer"));

    operator type *()
    {
        return value.get();
    }

    operator type &()
    {
        return *value;
    }

    bool load(handle src, bool);
};

}} // namespace PYBIND11_NAMESPACE::detail

#endif // NVCV_PYTHON_CUDA_BUFFER_HPP
