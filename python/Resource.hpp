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

#ifndef NVCV_PYTHON_RESOURCE_HPP
#define NVCV_PYTHON_RESOURCE_HPP

#include "Object.hpp"

#include <nvcv/detail/CudaFwd.h>
#include <pybind11/pybind11.h>

#include <memory>

// fwd declaration from driver_types.h
typedef struct CUevent_st *cudaEvent_t;

namespace nv::cvpy {
namespace py = pybind11;

class Stream;

enum LockMode
{
    LOCK_NONE      = 0,
    LOCK_READ      = 1,
    LOCK_WRITE     = 2,
    LOCK_READWRITE = LOCK_READ | LOCK_WRITE
};

class Resource : public virtual Object
{
public:
    ~Resource();

    static void Export(py::module &m);

    uint64_t id() const;

    void submitSync(Stream &stream, LockMode mode) const;
    void submitSignal(Stream &stream, LockMode mode) const;

    // Assumes GIL is locked (is in acquired state)
    void sync(LockMode mode) const;

    std::shared_ptr<Resource>       shared_from_this();
    std::shared_ptr<const Resource> shared_from_this() const;

protected:
    Resource();

    void doSubmitSync(Stream &stream, LockMode mode) const;

    // Assumes GIL is not locked (is in released state)
    void doSync(LockMode mode) const;

private:
    // To be overriden by children if they have their own requirements
    virtual void doBeforeSync(LockMode mode) const {};
    virtual void doBeforeSubmitSync(Stream &stream, LockMode mode) const {};
    virtual void doBeforeSubmitSignal(Stream &stream, LockMode mode) const {};

    uint64_t    m_id;
    cudaEvent_t m_readEvent, m_writeEvent;
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_RESOURCE_HPP
