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

#include "Definitions.hpp"

#include <core/HandleManager.hpp>
#include <core/HandleManagerImpl.hpp>

namespace priv = nv::cv::priv;

namespace {
class alignas(priv::kResourceAlignment) IObject
{
public:
    virtual int value() const = 0;
};

class Object : public IObject
{
public:
    explicit Object(int val)
        : m_value(val)
    {
    }

    virtual int value() const override
    {
        return m_value;
    }

private:
    int m_value;
};
} // namespace

TEST(HandleManager, wip_handle_generation_wraps_around)
{
    priv::HandleManager<IObject, Object> mgr("Object");

    mgr.setFixedSize(1);

    void   *h;
    Object *obj;
    std::tie(h, obj) = mgr.create<Object>(0);
    ASSERT_EQ(0, obj->value());
    ASSERT_EQ(obj, mgr.validate(h));

    void *origh = h;

    for (int i = 1; i < 16; ++i)
    {
        IObject *obj = mgr.validate(h);
        ASSERT_EQ(i - 1, obj->value());

        mgr.destroy(h);
        void *newh = mgr.create<Object>(i).first;
        ASSERT_NE(h, newh) << "Handle generation must be different";

        IObject *newobj = mgr.validate(newh);
        ASSERT_EQ(obj, newobj);
        ASSERT_EQ(i, newobj->value());

        h = newh;
    }

    mgr.destroy(h);
    std::tie(h, obj) = mgr.create<Object>(16);
    ASSERT_EQ(origh, h) << "Handle generation must wrapped around";
    IObject *iobj = mgr.validate(h);
    ASSERT_EQ(obj, iobj);
    ASSERT_EQ(16, iobj->value());

    mgr.destroy(h);
}

TEST(HandleManager, wip_destroy_already_destroyed)
{
    priv::HandleManager<IObject, Object> mgr("Object");

    void *h = mgr.create<Object>(0).first;
    ASSERT_TRUE(mgr.destroy(h));
    ASSERT_FALSE(mgr.destroy(h));
}

TEST(HandleManager, wip_destroy_invalid)
{
    priv::HandleManager<IObject, Object> mgr("Object");

    void *h = mgr.create<Object>(0).first;
    ASSERT_FALSE(mgr.destroy((void *)0x666));

    ASSERT_TRUE(mgr.destroy(h));
}

TEST(HandleManager, wip_validate_already_destroyed)
{
    priv::HandleManager<IObject, Object> mgr("Object");

    void *h = mgr.create<Object>(0).first;
    ASSERT_NE(nullptr, mgr.validate(h));

    ASSERT_TRUE(mgr.destroy(h));
    ASSERT_EQ(nullptr, mgr.validate(h));
}

TEST(HandleManager, wip_validate_invalid)
{
    priv::HandleManager<IObject, Object> mgr("Object");

    void *h = mgr.create<Object>(0).first;
    ASSERT_NE(nullptr, mgr.validate(h)); // just to have something being managed already

    ASSERT_EQ(nullptr, mgr.validate((void *)0x666));

    ASSERT_TRUE(mgr.destroy(h));
}
