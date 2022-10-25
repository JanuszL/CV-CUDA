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

#ifndef NVCV_PYTHON_CONTAINER_HPP
#define NVCV_PYTHON_CONTAINER_HPP

#include "Cache.hpp"
#include "Resource.hpp"

#include <memory>

namespace nv::cvpy {
namespace py = pybind11;

class Container
    : public Resource
    , public CacheItem
{
public:
    static void Export(py::module &m);

    std::shared_ptr<Container>       shared_from_this();
    std::shared_ptr<const Container> shared_from_this() const;

protected:
    Container() = default;
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_CONTAINER_HPP
