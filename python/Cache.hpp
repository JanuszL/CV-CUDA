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

#ifndef NVCV_PYTHON_CACHE_HPP
#define NVCV_PYTHON_CACHE_HPP

#include "Hash.hpp"
#include "Object.hpp"

#include <pybind11/pybind11.h>

#include <vector>

namespace nv::cvpy {

namespace py = pybind11;

class IKey
{
public:
    virtual ~IKey() = default;

    size_t hash() const;
    bool   operator==(const IKey &that) const;

private:
    virtual size_t doGetHash() const                 = 0;
    virtual bool   doIsEqual(const IKey &that) const = 0;
};

class CacheItem : public virtual Object
{
public:
    uint64_t id() const;

    virtual const IKey &key() const = 0;

    std::shared_ptr<CacheItem>       shared_from_this();
    std::shared_ptr<const CacheItem> shared_from_this() const;

    bool isInUse() const;

protected:
    CacheItem();

private:
    uint64_t m_id;
};

class Cache
{
public:
    static void Export(py::module &m);

    static Cache &Instance();

    void add(CacheItem &container);

    std::vector<std::shared_ptr<CacheItem>> fetch(const IKey &key) const;

    template<class T>
    std::vector<std::shared_ptr<T>> fetchAll() const
    {
        std::vector<std::shared_ptr<T>> out;

        doIterateThroughItems(
            [&out](CacheItem &item)
            {
                if (auto titem = std::dynamic_pointer_cast<T>(item.shared_from_this()))
                {
                    out.emplace_back(std::move(titem));
                }
            });
        return out;
    }

    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    Cache();

    void doIterateThroughItems(const std::function<void(CacheItem &item)> &fn) const;
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_CACHE_HPP
