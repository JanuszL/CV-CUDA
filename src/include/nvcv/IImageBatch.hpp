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

#ifndef NVCV_IIMAGEBATCH_HPP
#define NVCV_IIMAGEBATCH_HPP

#include "Image.hpp"
#include "ImageBatch.h"
#include "ImageBatchData.hpp"
#include "detail/Optional.hpp"

#include <iterator>

namespace nv { namespace cv {

class IImageBatch
{
public:
    virtual ~IImageBatch() = default;

    NVCVImageBatchHandle handle() const;

    int32_t capacity() const;
    int32_t numImages() const;

    const IImageBatchData *exportData(CUstream stream) const;

private:
    virtual NVCVImageBatchHandle doGetHandle() const = 0;

    // Only one leaf, we can use an optional for now.
    mutable detail::Optional<ImageBatchVarShapeDataPitchDevice> m_cacheData;
};

class IImageBatchVarShape : public IImageBatch
{
public:
    template<class IT>
    void pushBack(IT itBeg, IT itend);
    void pushBack(const IImage &img);
    void popBack(int32_t imgCount = 1);

    // For any invocable functor with zero parameters
    template<class F, class = decltype(std::declval<F>()())>
    void pushBack(F &&cb);

    void clear();

    Size2D      maxSize() const;
    ImageFormat uniqueFormat() const;

    const IImageBatchVarShapeData *exportData(CUstream stream) const;

    ImageWrapHandle operator[](ptrdiff_t n) const;

    class Iterator;

    using ConstIterator = Iterator;

    ConstIterator begin() const;
    ConstIterator end() const;

    ConstIterator cbegin() const;
    ConstIterator cend() const;
};

class IImageBatchVarShape::Iterator
{
public:
    using value_type        = ImageWrapHandle;
    using reference         = const value_type &;
    using pointer           = const value_type *;
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = ptrdiff_t;

    Iterator();
    Iterator(const Iterator &that);
    Iterator &operator=(const Iterator &that);

    reference operator*() const;
    Iterator  operator++(int);
    Iterator &operator++();
    pointer   operator->() const;

    bool operator==(const Iterator &that) const;
    bool operator!=(const Iterator &that) const;

private:
    const IImageBatchVarShape *m_batch;
    int                        m_curIndex;

    mutable detail::Optional<ImageWrapHandle> m_opImage;

    friend class IImageBatchVarShape;
    Iterator(const IImageBatchVarShape &batch, int32_t idxImage);
};

}} // namespace nv::cv

#include "detail/IImageBatchImpl.hpp"

#endif // NVCV_IIMAGEBATCH_HPP
