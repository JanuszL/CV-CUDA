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

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace nv { namespace cv {

class Shape
{
    using Data = std::array<int32_t, NVCV_TENSOR_MAX_NDIM>;

public:
    using value_type      = Data::value_type;
    using size_type       = int32_t;
    using reference       = Data::reference;
    using const_reference = Data::const_reference;
    using iterator        = Data::iterator;
    using const_iterator  = Data::const_iterator;

    Shape();
    Shape(const Shape &that);
    explicit Shape(int size);

    template<class IT>
    Shape(IT itbeg, IT itend);
    Shape(std::initializer_list<int32_t> shape);

    reference       operator[](int i);
    const_reference operator[](int i) const;

    int32_t size() const;
    bool    empty() const;

    Data::iterator begin();
    Data::iterator end();

    Data::const_iterator begin() const;
    Data::const_iterator end() const;

    Data::const_iterator cbegin() const;
    Data::const_iterator cend() const;

    bool operator==(const Shape &that) const;
    bool operator!=(const Shape &that) const;

    bool operator<(const Shape &that) const;

private:
    Data   m_data;
    int8_t m_size;
};

// Implementation

inline Shape::Shape()
    : m_size(0)
{
}

inline Shape::Shape(int size)
    : m_size(size)
{
    std::fill(this->begin(), this->end(), 0);
}

inline Shape::Shape(const Shape &that)
    : m_size(that.m_size)
{
    std::copy(that.begin(), that.end(), m_data.begin());
}

inline Shape::Shape(std::initializer_list<int32_t> shape)
    : Shape(shape.begin(), shape.end())
{
}

template<class IT>
inline Shape::Shape(IT itbeg, IT itend)
    : m_size(std::distance(itbeg, itend))
{
    assert(m_size <= (int32_t)m_data.size());
    std::copy(itbeg, itend, m_data.begin());
}

inline int32_t &Shape::operator[](int i)
{
    assert(0 <= i && i < m_size);
    return m_data[i];
}

inline const int32_t &Shape::operator[](int i) const
{
    assert(0 <= i && i < m_size);
    return m_data[i];
}

inline bool Shape::operator==(const Shape &that) const
{
    if (m_size == that.m_size)
    {
        return std::equal(this->begin(), this->end(), that.begin());
    }
    else
    {
        return false;
    }
}

inline bool Shape::operator!=(const Shape &that) const
{
    return !operator==(that);
}

inline bool Shape::operator<(const Shape &that) const
{
    return std::lexicographical_compare(this->begin(), this->end(), that.begin(), that.end());
}

inline int32_t Shape::size() const
{
    return m_size;
}

inline bool Shape::empty() const
{
    return m_size == 0;
}

inline auto Shape::begin() -> Data::iterator
{
    return m_data.begin();
}

inline auto Shape::end() -> Data::iterator
{
    return m_data.begin() + m_size;
}

inline auto Shape::begin() const -> Data::const_iterator
{
    return m_data.begin();
}

inline auto Shape::end() const -> Data::const_iterator
{
    return m_data.begin() + m_size;
}

inline auto Shape::cbegin() const -> Data::const_iterator
{
    return m_data.cbegin();
}

inline auto Shape::cend() const -> Data::const_iterator
{
    return m_data.cend() + m_size;
}

inline std::ostream &operator<<(std::ostream &out, const Shape &shape)
{
    if (shape.empty())
    {
        return out << "empty";
    }
    else
    {
        out << shape[0];
        for (int i = 0; i < shape.size(); ++i)
        {
            out << 'x' << shape[i];
        }
        return out;
    }
}

}} // namespace nv::cv

#endif // NVCV_SHAPE_HPP
