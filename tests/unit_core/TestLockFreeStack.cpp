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

#include "Definitions.hpp"

#include <core/LockFreeStack.hpp>

namespace priv = nv::cv::priv;

struct Node
{
    Node(int v = 0)
        : value(v){};
    int value;

    Node *next = nullptr;
};

TEST(LockFreeStack, wip_push)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    Node n[3];
    for (int i = 0; i < 3; ++i)
    {
        n[i].value = i;
        stack.push(n + i);
    }

    ASSERT_EQ(n + 2, stack.top());
    ASSERT_EQ(n + 1, n[2].next);
    ASSERT_EQ(n + 0, n[1].next);
    ASSERT_EQ(nullptr, n[0].next);

    ASSERT_EQ(n + 2, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(n + 1, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(n + 0, stack.pop());
    ASSERT_TRUE(stack.empty());
}

TEST(LockFreeStack, wip_push_stack)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    Node n(0);
    stack.push(&n);

    Node nn[3];
    for (int i = 0; i < 3; ++i)
    {
        nn[i].value = i;
        nn[i].next  = i + 1 < 3 ? &nn[i + 1] : nullptr;
    }

    stack.pushStack(nn, nn + 2);

    ASSERT_EQ(nn + 0, stack.top());
    ASSERT_EQ(nn + 1, nn[0].next);
    ASSERT_EQ(nn + 2, nn[1].next);
    ASSERT_EQ(&n, nn[2].next);
    ASSERT_EQ(nullptr, n.next);

    ASSERT_EQ(nn + 0, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(nn + 1, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(nn + 2, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(&n, stack.pop());
    ASSERT_TRUE(stack.empty());

    ASSERT_EQ(nullptr, stack.pop());
    ASSERT_TRUE(stack.empty());
}

TEST(LockFreeStack, wip_release)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    Node nn[3];
    for (int i = 0; i < 3; ++i)
    {
        nn[i].value = i;
        nn[i].next  = i + 1 < 3 ? &nn[i + 1] : nullptr;
    }

    stack.pushStack(nn, nn + 2);

    Node *h = stack.release();

    EXPECT_EQ(nullptr, stack.top());

    EXPECT_EQ(nn + 0, h);
    EXPECT_EQ(nn + 1, nn[0].next);
    EXPECT_EQ(nn + 2, nn[1].next);
    EXPECT_EQ(nullptr, nn[2].next);
}
