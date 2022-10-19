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

#ifndef NVCV_SHAPE_HPP
#define NVCV_SHAPE_HPP

#include <nvcv/TensorData.h>

#include <array>
#include <iostream>

namespace nv { namespace cv {

class Shape
{
    using Data = std::array<int32_t, NVCV_TENSOR_MAX_NDIM>;

public:
    using value_type = Data::value_type;

    int32_t       &operator[](int i);
    const int32_t &operator[](int i) const;

    int32_t size() const;

    Data::iterator begin();
    Data::iterator end();

    Data::const_iterator begin() const;
    Data::const_iterator end() const;

    Data::const_iterator cbegin() const;
    Data::const_iterator cend() const;

    bool operator==(const Shape &that) const;
    bool operator!=(const Shape &that) const;

    bool operator<(const Shape &that) const;

    // Must be public so that we can use implicit list initializers
    Data m_data;
};

// Implementation

inline int32_t &Shape::operator[](int i)
{
    assert(0 <= i && i < (int)m_data.size());
    return m_data[i];
}

inline const int32_t &Shape::operator[](int i) const
{
    assert(0 <= i && i < (int)m_data.size());
    return m_data[i];
}

inline bool Shape::operator==(const Shape &that) const
{
    return m_data == that.m_data;
}

inline bool Shape::operator!=(const Shape &that) const
{
    return m_data != that.m_data;
}

inline bool Shape::operator<(const Shape &that) const
{
    return m_data < that.m_data;
}

inline int32_t Shape::size() const
{
    return m_data.size();
}

inline auto Shape::begin() -> Data::iterator
{
    return m_data.begin();
}

inline auto Shape::end() -> Data::iterator
{
    return m_data.begin();
}

inline auto Shape::begin() const -> Data::const_iterator
{
    return m_data.begin();
}

inline auto Shape::end() const -> Data::const_iterator
{
    return m_data.begin();
}

inline auto Shape::cbegin() const -> Data::const_iterator
{
    return m_data.cbegin();
}

inline auto Shape::cend() const -> Data::const_iterator
{
    return m_data.cbegin();
}

inline std::ostream &operator<<(std::ostream &out, const Shape &shape)
{
    out << shape[0];
    for (int i = 1; i < shape.size(); ++i)
    {
        out << 'x' << shape[i];
    }
    return out;
}

}} // namespace nv::cv

#endif // NVCV_SHAPE_HPP
