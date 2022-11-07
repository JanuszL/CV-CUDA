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

#ifndef NVCV_TENSORSHAPE_HPP
#define NVCV_TENSORSHAPE_HPP

#include "Shape.hpp"
#include "TensorLayout.hpp"
#include "TensorShape.h"
#include "detail/Concepts.hpp"

namespace nv { namespace cv {

class TensorShape
{
public:
    using DimType                 = int64_t;
    using ShapeType               = Shape<DimType, NVCV_TENSOR_MAX_NDIM>;
    constexpr static int MAX_NDIM = ShapeType::MAX_NDIM;

    TensorShape() = default;

    TensorShape(ShapeType shape, TensorLayout layout)
        : m_shape(std::move(shape))
        , m_layout(std::move(layout))
    {
        if (m_layout != TensorLayout::NONE && m_shape.ndim() != m_layout.ndim())
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Layout dimensions must match shape dimensions");
        }
    }

    TensorShape(int size, TensorLayout layout)
        : TensorShape(ShapeType(size), std::move(layout))
    {
    }

    explicit TensorShape(TensorLayout layout)
        : TensorShape(layout.ndim(), std::move(layout))
    {
    }

    TensorShape(const DimType *data, int32_t size, TensorLayout layout)
        : TensorShape(ShapeType(data, size), std::move(layout))
    {
    }

    TensorShape(const DimType *data, int32_t size, const char *layout)
        : TensorShape(ShapeType(data, size), TensorLayout{layout})
    {
    }

    TensorShape(ShapeType shape, const char *layout)
        : TensorShape(std::move(shape), TensorLayout{layout})
    {
    }

    const ShapeType &shape() const
    {
        return m_shape;
    }

    const TensorLayout &layout() const
    {
        return m_layout;
    }

    const DimType &operator[](int i) const
    {
        return m_shape[i];
    }

    int ndim() const
    {
        return m_shape.ndim();
    }

    int size() const
    {
        return m_shape.size();
    }

    bool empty() const
    {
        return m_shape.empty();
    }

    bool operator==(const TensorShape &that) const
    {
        return std::tie(m_shape, m_layout) == std::tie(that.m_shape, that.m_layout);
    }

    bool operator!=(const TensorShape &that) const
    {
        return !(*this == that);
    }

    bool operator<(const TensorShape &that) const
    {
        return std::tie(m_shape, m_layout) < std::tie(that.m_shape, that.m_layout);
    }

    friend std::ostream &operator<<(std::ostream &out, const TensorShape &ts)
    {
        if (ts.m_layout == TensorLayout::NONE)
        {
            return out << ts.m_shape;
        }
        else
        {
            return out << ts.m_layout << '{' << ts.m_shape << '}';
        }
    }

private:
    ShapeType    m_shape;
    TensorLayout m_layout;
};

inline TensorShape Permute(const TensorShape &src, TensorLayout dstLayout)
{
    TensorShape::ShapeType dst(dstLayout.ndim());
    detail::CheckThrow(nvcvTensorShapePermute(src.layout(), &src[0], dstLayout, &dst[0]));

    return {std::move(dst), std::move(dstLayout)};
}

}} // namespace nv::cv

#endif // NVCV_TENSORSHAPE_HPP
