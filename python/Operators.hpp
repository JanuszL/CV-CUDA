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

#include "Container.hpp"

#include <pybind11/pybind11.h>

namespace nv::cvpy {

namespace py = ::pybind11;

void ExportOpReformat(py::module &m);
void ExportOpResize(py::module &m);
void ExportOpCustomCrop(py::module &m);
void ExportOpNormalize(py::module &m);
void ExportOpConvertTo(py::module &m);
void ExportOpPadAndStack(py::module &m);
void ExportOpCopyMakeBorder(py::module &m);
void ExportOpRotate(py::module &m);
void ExportOpErase(py::module &m);
void ExportOpGaussian(py::module &m);
void ExportOpMedianBlur(py::module &m);
void ExportOpLaplacian(py::module &m);
void ExportOpAverageBlur(py::module &m);
void ExportOpConv2D(py::module &m);
void ExportOpBilateralFilter(py::module &m);
void ExportOpCenterCrop(py::module &m);

// Helper class that serves as python-side operator class.
// OP: native operator class
// CTOR: ctor signature
template<class OP, class CTOR>
class PyOperator;

template<class OP, class... CTOR_ARGS>
class PyOperator<OP, void(CTOR_ARGS...)> : public Container
{
public:
    template<class... AA>
    void submit(AA &&...args)
    {
        m_op(std::forward<AA>(args)...);
    }

    const IKey &key() const override
    {
        return m_key;
    }

private:
    template<class OP2, class... AA2>
    friend std::shared_ptr<PyOperator<OP2, void(AA2...)>> CreateOperator(AA2 &&...args);

    PyOperator(const CTOR_ARGS &...args)
        : m_key{args...}
        , m_op{args...}
    {
    }

    class Key : public IKey
    {
    public:
        Key(const CTOR_ARGS &...args)
            : m_args{args...}
        {
        }

    private:
        size_t doGetHash() const override
        {
            return apply([](auto... args) { return ComputeHash(args...); }, m_args);
        }

        bool doIsEqual(const IKey &ithat) const override
        {
            auto &that = static_cast<const Key &>(ithat);
            return m_args == that.m_args;
        }

        std::tuple<CTOR_ARGS...> m_args;
    };

    // order is important
    Key m_key;
    OP  m_op;
};

// Returns an operator instance.
// Either gets it from the resource cache or creates one from scratch.
// When creationg, it'll add it to the cache.
template<class OP, class... AA>
std::shared_ptr<PyOperator<OP, void(AA...)>> CreateOperator(AA &&...args)
{
    using PyOP = PyOperator<OP, void(AA...)>;

    // Creates a key out of the operator's ctor parameters
    typename PyOP::Key key{args...};

    // Try to fetch it from cache
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(key);

    // None found?
    if (vcont.empty())
    {
        // Creates a new one
        auto op = std::shared_ptr<PyOP>(new PyOP(std::forward<AA>(args)...));
        // Adds to the resource cache
        Cache::Instance().add(*op);
        return op;
    }
    else
    {
        // Get the first one found in cache
        return std::static_pointer_cast<PyOP>(vcont[0]);
    }
}

} // namespace nv::cvpy
