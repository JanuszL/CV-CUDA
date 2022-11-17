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

#include "Stream.hpp"

#include "Assert.hpp"
#include "Cache.hpp"
#include "CheckError.hpp"
#include "PyUtil.hpp"
#include "String.hpp"

#include <pybind11/operators.h>

#include <stack>

namespace nv::cvpy {

// Here we define the representation of external cuda streams.
// It defines pybind11's type casters from the python object
// to the corresponding ExternalStream<E>.

// Defines each external stream represetation we support.
enum ExternalStreamType
{
    VOIDP,
    INT,
    TORCH,
    NUMBA,
};

template<ExternalStreamType E>
class ExternalStream : public IExternalStream
{
public:
    void setCudaStream(cudaStream_t cudaStream, py::object obj)
    {
        m_cudaStream = cudaStream;
        m_wrappedObj = std::move(obj);
    }

    virtual cudaStream_t handle() const override
    {
        return m_cudaStream;
    }

    virtual py::object wrappedObject() const override
    {
        return m_wrappedObj;
    }

private:
    cudaStream_t m_cudaStream;
    py::object   m_wrappedObj;
};

} // namespace nv::cvpy

namespace PYBIND11_NAMESPACE { namespace detail {

using namespace std::literals;
namespace cvpy = nv::cvpy;

template<>
struct type_caster<cvpy::ExternalStream<cvpy::VOIDP>>
{
    PYBIND11_TYPE_CASTER(cvpy::ExternalStream<cvpy::VOIDP>, const_name("ctypes.c_void_p"));

    bool load(handle src, bool)
    {
        std::string strType = cvpy::GetFullyQualifiedName(src);

        if (strType != "ctypes.c_void_p")
        {
            return false;
        }

        buffer_info info = ::pybind11::cast<buffer>(src).request();

        NVCV_ASSERT(info.itemsize == sizeof(void *));

        void *data = *reinterpret_cast<void **>(info.ptr);

        value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(std::move(src)));
        return true;
    }
};

template<>
struct type_caster<cvpy::ExternalStream<cvpy::INT>>
{
    PYBIND11_TYPE_CASTER(cvpy::ExternalStream<cvpy::INT>, const_name("int"));

    bool load(handle src, bool)
    {
        try
        {
            // TODO: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = src.cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(std::move(src)));
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};

template<>
struct type_caster<cvpy::ExternalStream<cvpy::TORCH>>
{
    PYBIND11_TYPE_CASTER(cvpy::ExternalStream<cvpy::TORCH>, const_name("torch.cuda.Stream"));

    bool load(handle src, bool)
    {
        std::string strType = cvpy::GetFullyQualifiedName(src);

        if (strType != "torch.cuda.streams.Stream" && strType != "torch.cuda.streams.ExternalStream")
        {
            return false;
        }

        try
        {
            // TODO: don't know how to test if a python object
            // is convertible to a type without exceptions.
            intptr_t data = src.attr("cuda_stream").cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(std::move(src)));
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};

template<>
struct type_caster<cvpy::ExternalStream<cvpy::NUMBA>>
{
    PYBIND11_TYPE_CASTER(cvpy::ExternalStream<cvpy::NUMBA>, const_name("numba.cuda.Stream"));

    bool load(handle src, bool)
    {
        std::string strType = cvpy::GetFullyQualifiedName(src);

        if (strType != "numba.cuda.cudadrv.driver.Stream")
        {
            return false;
        }

        try
        {
            // NUMBA cuda stream can be converted to ints, which is the cudaStream handle.
            intptr_t data = src.cast<intptr_t>();
            value.setCudaStream(reinterpret_cast<cudaStream_t>(data), ::pybind11::cast<object>(std::move(src)));
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};

}} // namespace PYBIND11_NAMESPACE::detail

namespace nv::cvpy {

// In terms of caching, all streams are the same.
// Any stream in the cache can be fetched and used.
size_t Stream::Key::doGetHash() const
{
    return 0;
}

bool Stream::Key::doIsEqual(const IKey &that) const
{
    return true;
}

std::shared_ptr<Stream> Stream::Create()
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Stream::Key{});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Stream> stream(new Stream());
        Cache::Instance().add(*stream);
        return stream;
    }
    else
    {
        // Get the first one
        return std::static_pointer_cast<Stream>(vcont[0]);
    }
}

Stream::Stream()
    : m_owns(true)
{
    CheckThrow(cudaStreamCreate(&m_handle));
}

Stream::Stream(IExternalStream &extStream)
    : m_owns(false)
    , m_handle(extStream.handle())
    , m_wrappedObj(std::move(extStream.wrappedObject()))
{
    unsigned int flags;
    if (cudaStreamGetFlags(m_handle, &flags) != cudaSuccess)
    {
        throw std::runtime_error("Invalid cuda stream");
    }
}

Stream::~Stream()
{
    if (m_owns)
    {
        CheckLog(cudaStreamDestroy(m_handle));
    }
}

std::shared_ptr<Stream> Stream::shared_from_this()
{
    return std::dynamic_pointer_cast<Stream>(Object::shared_from_this());
}

std::shared_ptr<const Stream> Stream::shared_from_this() const
{
    return std::dynamic_pointer_cast<const Stream>(Object::shared_from_this());
}

cudaStream_t Stream::handle() const
{
    return m_handle;
}

intptr_t Stream::pyhandle() const
{
    return reinterpret_cast<intptr_t>(m_handle);
}

void Stream::sync()
{
    py::gil_scoped_release release;

    CheckThrow(cudaStreamSynchronize(m_handle));
}

static std::stack<std::weak_ptr<Stream>> g_streamStack;
static std::weak_ptr<Stream>             g_stream;

Stream &Stream::Current()
{
    NVCV_ASSERT(!g_streamStack.empty());
    auto defStream = g_streamStack.top().lock();
    NVCV_ASSERT(defStream);
    return *defStream;
}

void Stream::activate()
{
    g_streamStack.push(this->shared_from_this());
}

void Stream::deactivate(py::object exc_type, py::object exc_value, py::object exc_tb)
{
    g_streamStack.pop();
}

void Stream::holdResources(std::vector<std::shared_ptr<const Resource>> usedResources)
{
    struct HostFunctionClosure
    {
        // Also hold the stream reference so that it isn't destroyed before the processing is done.
        std::shared_ptr<const Stream>                stream;
        std::vector<std::shared_ptr<const Resource>> resources;
    };

    auto closure = std::make_unique<HostFunctionClosure>();

    closure->stream    = this->shared_from_this();
    closure->resources = std::move(usedResources);

    auto fn = [](cudaStream_t stream, cudaError_t error, void *userData) -> void
    {
        auto *pclosure = reinterpret_cast<HostFunctionClosure *>(userData);
        delete pclosure;
    };

    CheckThrow(cudaStreamAddCallback(m_handle, fn, closure.get(), 0));

    closure.release();
}

std::ostream &operator<<(std::ostream &out, const Stream &stream)
{
    return out << "<nvcv.cuda.Stream id=" << stream.id() << " handle=" << stream.handle() << '>';
}

template<ExternalStreamType E>
static void ExportExternalStream(py::module &m)
{
    m.def("as_stream", [](ExternalStream<E> extStream) { return std::shared_ptr<Stream>(new Stream(extStream)); });
}

void Stream::Export(py::module &m)
{
    py::class_<Stream, std::shared_ptr<Stream>> stream(m, "Stream");

    stream.def_property_readonly_static("current", [](py::object) { return Current().shared_from_this(); })
        .def(py::init(&Stream::Create));

    // Create the global stream object. It'll be destroyed when
    // python module is deinitialized.
    auto globalStream = Stream::Create();
    g_streamStack.push(globalStream);
    g_stream = globalStream;

    stream.attr("default") = g_stream.lock();

    // Order from most specific to less specific
    ExportExternalStream<TORCH>(m);
    ExportExternalStream<NUMBA>(m);
    ExportExternalStream<VOIDP>(m);
    ExportExternalStream<INT>(m);

    stream.def("__enter__", &Stream::activate)
        .def("__exit__", &Stream::deactivate)
        .def("sync", &Stream::sync)
        .def("__int__", &Stream::pyhandle)
        .def("__repr__", &ToString<Stream>)
        .def_property_readonly("handle", &Stream::pyhandle)
        .def_property_readonly("id", &Stream::id);

    // Make sure all streams we've created are synced when script ends.
    // Also make cleanup hold the globalStream reference during script execution.
    RegisterCleanup(m,
                    [globalStream]()
                    {
                        for (std::shared_ptr<Stream> stream : Cache::Instance().fetchAll<Stream>())
                        {
                            stream->sync();
                        }
                    });
}

} // namespace nv::cvpy
