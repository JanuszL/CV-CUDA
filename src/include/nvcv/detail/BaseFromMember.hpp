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

#ifndef NVCV_DETAIL_BASE_FROM_MEMBER_HPP
#define NVCV_DETAIL_BASE_FROM_MEMBER_HPP

#include <utility>

namespace nv { namespace cv { namespace detail {

// Base-from-member idiom
// Ref: https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Base-from-Member

/* Needed when we have to pass a member variable to a base class.
   To make sure the member is fully constructed, it must be defined as
   a base class, coming *before* the base that needs it. C++ rules
   guarantees that base classes are constructed in definition order.
   If there are multiple members with the same type, you must give them
   different IDs.

   Ex:
     struct Bar{};

     struct Foo
     {
         Foo(Bar &, Bar * =nullptr);
     };

     struct A
        : BaseFromMember<Bar>
        , Foo
     {
        using MemberBar = BaseFromMember<Bar>;

        A()
            : Foo(MemberBar::member)
        {
        }
     };

     struct B
        : BaseFromMember<Bar,0>
        , BaseFromMember<Bar,1>
        , Foo
     {
        using MemberBar0 = BaseFromMember<Bar,0>;
        using MemberBar1 = BaseFromMember<Bar,1>;

        A()
            : Foo(MemberBar0::member, MemberBar1::member)
        {
        }
     };
*/

template<class T, int ID = 0>
class BaseFromMember
{
public:
    T member;
};

template<class T, int ID>
class BaseFromMember<T &, ID>
{
public:
    BaseFromMember(T &m)
        : member(m)
    {
    }

    T &member;
};

}}} // namespace nv::cv::detail

#endif // NVCV_DETAIL_BASE_FROM_MEMBER_HPP
