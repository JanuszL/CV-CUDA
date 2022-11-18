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

namespace detail {
struct GetImageHandle;
}

class IImageBatch
{
public:
    virtual ~IImageBatch() = default;

    NVCVImageBatchHandle handle() const;
    ImageFormat          format() const;
    IAllocator          &alloc() const;

    int32_t capacity() const;
    int32_t numImages() const;

    const IImageBatchData *exportData(CUstream stream) const;

protected:
    virtual const IImageBatchData *doExportData(CUstream stream) const = 0;

private:
    virtual NVCVImageBatchHandle doGetHandle() const = 0;

    virtual ImageFormat doGetFormat() const    = 0;
    virtual int32_t     doGetCapacity() const  = 0;
    virtual int32_t     doGetNumImages() const = 0;

    virtual IAllocator &doGetAlloc() const = 0;
};

class IImageBatchVarShape : public virtual IImageBatch
{
public:
    template<class IT>
    void pushBack(IT itBeg, IT itend);
    void pushBack(const IImage &img);
    void popBack(int32_t imgCount = 1);

    // For any reference wrapper of any other accepted type
    template<class F, class = decltype(std::declval<detail::GetImageHandle>()(std::declval<F>()()))>
    void pushBack(F &&cv);

    void clear();

    const IImageBatchVarShapeData *exportData(CUstream stream) const;

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

// IImageBatch implementation

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

inline int32_t IImageBatch::numImages() const
{
    int32_t s = doGetNumImages();
    assert(s >= 0);
    assert(s <= this->capacity());
    return s;
}

inline const IImageBatchData *IImageBatch::exportData(CUstream stream) const
{
    return doExportData(stream);
}

// IImageBatchVarShape implementation

inline auto IImageBatchVarShape::begin() const -> ConstIterator
{
    return ConstIterator(*this, 0);
}

inline auto IImageBatchVarShape::end() const -> ConstIterator
{
    return ConstIterator(*this, this->numImages());
}

inline auto IImageBatchVarShape::cbegin() const -> ConstIterator
{
    return this->begin();
}

inline auto IImageBatchVarShape::cend() const -> ConstIterator
{
    return this->end();
}

inline const IImageBatchVarShapeData *IImageBatchVarShape::exportData(CUstream stream) const
{
    return static_cast<const IImageBatchVarShapeData *>(doExportData(stream));
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
    if (m_batch == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Iterator doesn't point to an image batch object");
    }
    if (m_curIndex >= m_batch->numImages())
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Iterator points to an invalid image in the image batch");
    }

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
        if (ptr == nullptr)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image must not be NULL");
        }
        return ptr->handle();
    }

    NVCVImageHandle operator()(const IImage &img) const
    {
        return img.handle();
    }

    NVCVImageHandle operator()(NVCVImageHandle h) const
    {
        if (h == nullptr)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image handle must not be NULL");
        }
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

// Functor must return a type that can be converted to an image handle
template<class F, class SFINAE>
void IImageBatchVarShape::pushBack(F &&cb)
{
    auto cb2 = [cb = std::move(cb)]() mutable -> NVCVImageHandle
    {
        if (auto img = cb())
        {
            detail::GetImageHandle imgHandle;
            return imgHandle(img);
        }
        else
        {
            return nullptr;
        }
    };

    doPushBack(std::ref(cb2));
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
