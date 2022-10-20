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

#ifndef NVCV_PYTHON_STREAM_HPP
#define NVCV_PYTHON_STREAM_HPP

#include "Cache.hpp"
#include "Object.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace nv::cvpy {

class IExternalStream
{
public:
    virtual cudaStream_t handle() const        = 0;
    virtual py::object   wrappedObject() const = 0;
};

class Stream : public CacheItem
{
public:
    static void Export(py::module &m);

    static Stream &Current();

    static std::shared_ptr<Stream> Create();

    ~Stream();

    std::shared_ptr<Stream>       shared_from_this();
    std::shared_ptr<const Stream> shared_from_this() const;

    void activate();
    void deactivate(py::object exc_type, py::object exc_value, py::object exc_tb);

    void         sync();
    cudaStream_t handle() const;

    // Returns the cuda handle in python
    intptr_t pyhandle() const;

    Stream(IExternalStream &extStream);

    friend std::ostream &operator<<(std::ostream &out, const Stream &stream);

private:
    Stream(Stream &&) = delete;
    Stream();

    class Key final : public IKey
    {
    private:
        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        static Key key;
        return key;
    }

    bool         m_owns;
    cudaStream_t m_handle;
    py::object   m_wrappedObj;
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_STREAM_HPP
