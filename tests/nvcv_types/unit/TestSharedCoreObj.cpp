/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"

#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv_types/priv/Context.hpp>
#include <nvcv_types/priv/DefaultAllocator.hpp>
#include <nvcv_types/priv/Exception.hpp>
#include <nvcv_types/priv/Image.hpp>
#include <nvcv_types/priv/ImageManager.hpp>
#include <nvcv_types/priv/SharedCoreObj.hpp>

TEST(SharedCoreObjTest, ConstructFromPointer)
{
    nvcv::Size2D          size{640, 480};
    nvcv::ImageFormat     fmt = nvcv::FMT_RGBA8;
    NVCVImageRequirements reqs;
    nvcv::detail::CheckThrow(nvcvImageCalcRequirements(size.w, size.h, fmt, 256, 16, &reqs));
    auto &alloc = nvcv::priv::GetDefaultAllocator();

    auto h = nvcv::priv::CreateCoreObject<nvcv::priv::Image>(reqs, alloc);
    EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 1);
    {
        nvcv::priv::SharedCoreObj<nvcv::priv::Image> s = nvcv::priv::ToSharedObj<nvcv::priv::Image>(h);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 2);
        EXPECT_EQ(s->handle(), h);
    }
    EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 1);
    EXPECT_EQ(nvcv::priv::CoreObjectDecRef(h), 0);
}
