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

namespace nv::cvpy {

std::shared_ptr<Container> Container::shared_from_this()
{
    return std::static_pointer_cast<Container>(Resource::shared_from_this());
}

std::shared_ptr<const Container> Container::shared_from_this() const
{
    return std::static_pointer_cast<const Container>(Resource::shared_from_this());
}

void Container::Export(py::module &m)
{
    py::class_<Container, std::shared_ptr<Container>, Resource> cont(m, "Container");
}

} // namespace nv::cvpy
