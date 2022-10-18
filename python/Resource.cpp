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

#include "Resource.hpp"

#include "Assert.hpp"
#include "CheckError.hpp"
#include "Stream.hpp"

#include <iostream>

namespace nv::cvpy {

Resource::Resource()
{
    static uint64_t idnext = 0;

    m_id = idnext++;

    m_readEvent = m_writeEvent = nullptr;
    try
    {
        CheckThrow(cudaEventCreateWithFlags(&m_readEvent, cudaEventDisableTiming));
        CheckThrow(cudaEventCreateWithFlags(&m_writeEvent, cudaEventDisableTiming));
    }
    catch (...)
    {
        cudaEventDestroy(m_readEvent);
        cudaEventDestroy(m_writeEvent);
        throw;
    }
}

Resource::~Resource()
{
    cudaEventDestroy(m_readEvent);
    cudaEventDestroy(m_writeEvent);
}

uint64_t Resource::id() const
{
    return m_id;
}

void Resource::submitSignal(Stream &stream, LockMode mode) const
{
    doBeforeSubmitSignal(stream, mode);

    if (mode & LOCK_READ)
    {
        CheckThrow(cudaEventRecord(m_readEvent, stream.handle()));
    }
    if (mode & LOCK_WRITE)
    {
        CheckThrow(cudaEventRecord(m_writeEvent, stream.handle()));
    }
}

void Resource::submitSync(Stream &stream, LockMode mode) const
{
    doBeforeSubmitSync(stream, mode);

    doSubmitSync(stream, mode);
}

void Resource::doSubmitSync(Stream &stream, LockMode mode) const
{
    if (mode & LOCK_READ)
    {
        CheckThrow(cudaStreamWaitEvent(stream.handle(), m_writeEvent));
    }
    if (mode & LOCK_WRITE)
    {
        CheckThrow(cudaStreamWaitEvent(stream.handle(), m_writeEvent));
        CheckThrow(cudaStreamWaitEvent(stream.handle(), m_readEvent));
    }
}

void Resource::sync(LockMode mode) const
{
    py::gil_scoped_release release;

    doBeforeSync(mode);

    doSync(mode);
}

void Resource::doSync(LockMode mode) const
{
    NVCV_ASSERT(PyGILState_Check() == 0);

    if (mode & LOCK_READ)
    {
        CheckThrow(cudaEventSynchronize(m_writeEvent));
    }
    if (mode & LOCK_WRITE)
    {
        CheckThrow(cudaEventSynchronize(m_writeEvent));
        CheckThrow(cudaEventSynchronize(m_readEvent));
    }
}

std::shared_ptr<Resource> Resource::shared_from_this()
{
    return std::dynamic_pointer_cast<Resource>(Object::shared_from_this());
}

std::shared_ptr<const Resource> Resource::shared_from_this() const
{
    return std::dynamic_pointer_cast<const Resource>(Object::shared_from_this());
}

void Resource::Export(py::module &m)
{
    py::class_<Resource, std::shared_ptr<Resource>> resource(m, "Resource");

    resource.def_property_readonly("id", &Resource::id, "Unique resource instance identifier");
}

} // namespace nv::cvpy
