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

#include "IImageBatchData.hpp"
#include "Image.hpp"
#include "ImageBatch.h"
#include "detail/CudaFwd.h"
#include "detail/Optional.hpp"

#include <cassert>
#include <functional>
#include <iterator>

namespace nv { namespace cv {

class IImageBatch
{
public:
    virtual ~IImageBatch() = default;

    NVCVImageBatchHandle handle() const;
    ImageFormat          format() const;
    IAllocator          &alloc() const;

    int32_t capacity() const;
    int32_t size() const;

    const IImageBatchData *exportData(CUstream stream) const;

private:
    virtual NVCVImageBatchHandle doGetHandle() const = 0;

    virtual ImageFormat doGetFormat() const   = 0;
    virtual int32_t     doGetCapacity() const = 0;
    virtual int32_t     doGetSize() const     = 0;

    virtual IAllocator &doGetAlloc() const = 0;

    virtual const IImageBatchData *doExportData(CUstream stream) const = 0;
};

class IImageBatchVarShape : public virtual IImageBatch
{
public:
    template<class IT>
    void pushBack(IT itBeg, IT itend);
    void pushBack(const IImage &img);
    void popBack(int32_t imgCount = 1);

    void clear();

    class Iterator
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

    using ConstIterator = Iterator;

    ConstIterator begin() const;
    ConstIterator end() const;

    ConstIterator cbegin() const;
    ConstIterator cend() const;

private:
    virtual void doPushBack(std::function<NVCVImageHandle()> &&cb) = 0;
    virtual void doPushBack(const IImage &img)                     = 0;
    virtual void doPopBack(int32_t imgCount)                       = 0;
    virtual void doClear()                                         = 0;

    virtual NVCVImageHandle doGetImage(int32_t idx) const = 0;
};

// Implementation

inline NVCVImageBatchHandle IImageBatch::handle() const
{
    return doGetHandle();
}

inline ImageFormat IImageBatch::format() const
{
    return doGetFormat();
}

inline IAllocator &IImageBatch::alloc() const
{
    return doGetAlloc();
}

inline int32_t IImageBatch::capacity() const
{
    int32_t c = doGetCapacity();
    assert(c >= 0);
    return c;
}

inline int32_t IImageBatch::size() const
{
    int32_t s = doGetSize();
    assert(s >= 0);
    assert(s <= this->capacity());
    return s;
}

inline const IImageBatchData *IImageBatch::exportData(CUstream stream) const
{
    return doExportData(stream);
}

inline auto IImageBatchVarShape::begin() const -> ConstIterator
{
    return ConstIterator(*this, 0);
}

inline auto IImageBatchVarShape::end() const -> ConstIterator
{
    return ConstIterator(*this, this->size());
}

inline auto IImageBatchVarShape::cbegin() const -> ConstIterator
{
    return this->begin();
}

inline auto IImageBatchVarShape::cend() const -> ConstIterator
{
    return this->end();
}

inline IImageBatchVarShape::Iterator::Iterator(const IImageBatchVarShape &batch, int32_t idxImage)
    : m_batch(&batch)
    , m_curIndex(idxImage)
{
}

inline IImageBatchVarShape::Iterator::Iterator()
    : m_batch(nullptr)
    , m_curIndex(0)
{
}

inline IImageBatchVarShape::Iterator::Iterator(const Iterator &that)
    : m_batch(that.m_batch)
    , m_curIndex(that.m_curIndex)
{
}

inline auto IImageBatchVarShape::Iterator::operator=(const Iterator &that) -> Iterator &
{
    if (this != &that)
    {
        m_batch    = that.m_batch;
        m_curIndex = that.m_curIndex;
    }
    return *this;
}

inline auto IImageBatchVarShape::Iterator::operator*() const -> reference
{
    assert(m_batch != nullptr);
    assert(m_curIndex < m_batch->size());

    if (!m_opImage)
    {
        m_opImage.emplace(m_batch->doGetImage(m_curIndex));
    }
    return *m_opImage;
}

inline auto IImageBatchVarShape::Iterator::operator->() const -> pointer
{
    return &*(*this);
}

inline auto IImageBatchVarShape::Iterator::operator++(int) -> Iterator
{
    Iterator cur(*this);
    ++(*this);
    return cur;
}

inline auto IImageBatchVarShape::Iterator::operator++() -> Iterator &
{
    ++m_curIndex;
    m_opImage.reset();
    return *this;
}

inline bool IImageBatchVarShape::Iterator::operator==(const Iterator &that) const
{
    if (m_batch == nullptr && that.m_batch == nullptr)
    {
        return true;
    }
    else if (m_batch == that.m_batch)
    {
        return m_curIndex == that.m_curIndex;
    }
    else
    {
        return false;
    }
}

inline bool IImageBatchVarShape::Iterator::operator!=(const Iterator &that) const
{
    return !(*this == that);
}

namespace detail {
struct GetImageHandle
{
    // For any pointer-like type
    template<class T,
             class = typename std::enable_if<std::is_same<
                 NVCVImageHandle, typename std::decay<decltype(std::declval<T>()->handle())>::type>::value>::type>
    NVCVImageHandle operator()(const T &ptr) const
    {
        assert(ptr != nullptr);
        return ptr->handle();
    }

    NVCVImageHandle operator()(const IImage &img) const
    {
        return img.handle();
    }

    NVCVImageHandle operator()(NVCVImageHandle h) const
    {
        assert(h != nullptr);
        return h;
    }

    // For any reference wrapper of any other accepted type
    template<class T, class = decltype(std::declval<GetImageHandle>()(std::declval<T>()))>
    NVCVImageHandle operator()(const std::reference_wrapper<T> &h) const
    {
        return h.get().handle();
    }
};
} // namespace detail

template<class IT>
void IImageBatchVarShape::pushBack(IT itBeg, IT itEnd)
{
    auto cb = [it = itBeg, &itEnd]() mutable -> NVCVImageHandle
    {
        if (it == itEnd)
        {
            return nullptr;
        }

        detail::GetImageHandle imgHandle;
        return imgHandle(*it++);
    };

    // constructing a std::function with a reference_wrapper is guaranteed
    // not to allocate memory from heap.
    doPushBack(std::ref(cb));
}

inline void IImageBatchVarShape::pushBack(const IImage &img)
{
    doPushBack(img);
}

inline void IImageBatchVarShape::popBack(int32_t imgCount)
{
    doPopBack(imgCount);
}

inline void IImageBatchVarShape::clear()
{
    doClear();
}

}} // namespace nv::cv

#endif // NVCV_IIMAGEBATCH_HPP
