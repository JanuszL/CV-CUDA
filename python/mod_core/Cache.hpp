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

#ifndef NVCV_PYTHON_CACHE_HPP
#define NVCV_PYTHON_CACHE_HPP

#include "Object.hpp"

#include <common/Hash.hpp>
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

class PYBIND11_EXPORT CacheItem : public virtual Object
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

class PYBIND11_EXPORT Cache
{
public:
    static void Export(py::module &m);

    static Cache &Instance();

    void add(CacheItem &container);
    void removeAllNotInUseMatching(const IKey &key);

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
