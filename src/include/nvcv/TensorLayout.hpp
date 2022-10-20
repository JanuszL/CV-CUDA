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

#ifndef NVCV_TENSOR_LAYOUT_HPP
#define NVCV_TENSOR_LAYOUT_HPP

#include "TensorLayout.h"
#include "detail/CheckError.hpp"
#include "detail/Concepts.hpp"

#include <cassert>
#include <iostream>

inline bool operator==(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs, rhs) == 0;
}

inline bool operator!=(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return !operator==(lhs, rhs);
}

inline bool operator<(const NVCVTensorLayout &lhs, const NVCVTensorLayout &rhs)
{
    return nvcvTensorLayoutCompare(lhs, rhs) < 0;
}

inline std::ostream &operator<<(std::ostream &out, const NVCVTensorLayout &layout)
{
    return out << nvcvTensorLayoutGetName(&layout);
}

namespace nv { namespace cv {

enum TensorLabel : char
{
    LABEL_BATCH   = NVCV_TLABEL_BATCH,
    LABEL_CHANNEL = NVCV_TLABEL_CHANNEL,
    LABEL_FRAME   = NVCV_TLABEL_FRAME,
    LABEL_DEPTH   = NVCV_TLABEL_DEPTH,
    LABEL_HEIGHT  = NVCV_TLABEL_HEIGHT,
    LABEL_WIDTH   = NVCV_TLABEL_WIDTH
};

class TensorLayout final
{
public:
    using const_iterator = const char *;
    using iterator       = const_iterator;
    using value_type     = char;

    TensorLayout() = default;

    explicit constexpr TensorLayout(const NVCVTensorLayout &layout)
        : m_layout(layout)
    {
    }

    explicit TensorLayout(const char *descr)
    {
        detail::CheckThrow(nvcvTensorLayoutMake(descr, &m_layout));
    }

    template<class IT, class = detail::IsRandomAccessIterator<IT>>
    explicit TensorLayout(IT itbeg, IT itend)
    {
        detail::CheckThrow(nvcvTensorLayoutMakeRange(&*itbeg, &*itend, &m_layout));
    }

    constexpr char operator[](int idx) const;
    constexpr int  ndim() const;

    int find(char dimLabel, int start = 0) const;

    bool startsWith(const TensorLayout &test) const
    {
        return nvcvTensorLayoutStartsWith(m_layout, test.m_layout) != 0;
    }

    bool endsWith(const TensorLayout &test) const
    {
        return nvcvTensorLayoutEndsWith(m_layout, test.m_layout) != 0;
    }

    TensorLayout subRange(int beg, int end) const
    {
        TensorLayout out;
        NVCVStatus   st = nvcvTensorLayoutMakeSubRange(m_layout, beg, end, &out.m_layout);
        (void)st;
        assert(st == NVCV_SUCCESS);
        return out;
    }

    TensorLayout first(int n) const
    {
        TensorLayout out;
        NVCVStatus   st = nvcvTensorLayoutMakeFirst(m_layout, n, &out.m_layout);
        (void)st;
        assert(st == NVCV_SUCCESS);
        return out;
    }

    TensorLayout last(int n) const
    {
        TensorLayout out;
        NVCVStatus   st = nvcvTensorLayoutMakeLast(m_layout, n, &out.m_layout);
        (void)st;
        assert(st == NVCV_SUCCESS);
        return out;
    }

    bool operator==(const TensorLayout &that) const;
    bool operator!=(const TensorLayout &that) const;
    bool operator<(const TensorLayout &that) const;

    constexpr const_iterator begin() const;
    constexpr const_iterator end() const;
    constexpr const_iterator cbegin() const;
    constexpr const_iterator cend() const;

    constexpr operator const NVCVTensorLayout &() const;

    friend std::ostream &operator<<(std::ostream &out, const TensorLayout &that);

    // Public so that class is trivial but still the
    // implicit ctors do the right thing
    NVCVTensorLayout m_layout;

#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) static const TensorLayout LAYOUT;

    NVCV_DETAIL_DEF_TLAYOUT(NONE)
#include "TensorLayoutDef.inc"
#undef NVCV_DETAIL_DEF_TLAYOUT
};

#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) constexpr const TensorLayout TensorLayout::LAYOUT{NVCV_TENSOR_##LAYOUT};
NVCV_DETAIL_DEF_TLAYOUT(NONE)
#include "TensorLayoutDef.inc"
#undef NVCV_DETAIL_DEF_TLAYOUT

constexpr char TensorLayout::operator[](int idx) const
{
    return nvcvTensorLayoutGetLabel(m_layout, idx);
}

constexpr int TensorLayout::ndim() const
{
    return nvcvTensorLayoutGetNumDim(m_layout);
}

inline int TensorLayout::find(char dimLabel, int start) const
{
    return nvcvTensorLayoutFindDimIndex(m_layout, dimLabel, start);
}

constexpr TensorLayout::operator const NVCVTensorLayout &() const
{
    return m_layout;
}

inline bool TensorLayout::operator==(const TensorLayout &that) const
{
    return m_layout == that.m_layout;
}

inline bool TensorLayout::operator!=(const TensorLayout &that) const
{
    return !operator==(that);
}

inline bool TensorLayout::operator<(const TensorLayout &that) const
{
    return m_layout < that.m_layout;
}

constexpr auto TensorLayout::begin() const -> const_iterator
{
    return nvcvTensorLayoutGetName(&m_layout);
}

constexpr inline auto TensorLayout::end() const -> const_iterator
{
    return this->begin() + this->ndim();
}

constexpr auto TensorLayout::cbegin() const -> const_iterator
{
    return this->begin();
}

constexpr auto TensorLayout::cend() const -> const_iterator
{
    return this->end();
}

inline std::ostream &operator<<(std::ostream &out, const TensorLayout &that)
{
    return out << that.m_layout;
}

}} // namespace nv::cv

#endif // NVCV_TENSOR_LAYOUT_HPP
