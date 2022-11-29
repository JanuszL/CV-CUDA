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

#include "ImageBatch.hpp"

#include "Assert.hpp"
#include "Image.hpp"

namespace nv::cvpy {

size_t ImageBatchVarShape::Key::doGetHash() const
{
    return ComputeHash(m_capacity);
}

bool ImageBatchVarShape::Key::doIsEqual(const IKey &ithat) const
{
    auto &that = static_cast<const Key &>(ithat);
    return m_capacity == that.m_capacity;
}

std::shared_ptr<ImageBatchVarShape> ImageBatchVarShape::Create(int capacity)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{capacity});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<ImageBatchVarShape> batch(new ImageBatchVarShape(capacity));
        Cache::Instance().add(*batch);
        return batch;
    }
    else
    {
        // Get the first one
        auto batch = std::static_pointer_cast<ImageBatchVarShape>(vcont[0]);
        batch->clear(); // make sure it's in pristine state
        return batch;
    }
}

ImageBatchVarShape::ImageBatchVarShape(int capacity)
    : m_key(capacity)
    , m_impl(capacity)
{
    m_list.reserve(capacity);
}

const cv::ImageBatchVarShape &ImageBatchVarShape::impl() const
{
    return m_impl;
}

cv::ImageBatchVarShape &ImageBatchVarShape::impl()
{
    return m_impl;
}

py::object ImageBatchVarShape::uniqueFormat() const
{
    cv::ImageFormat fmt = m_impl.uniqueFormat();
    if (fmt)
    {
        return py::cast(fmt);
    }
    else
    {
        return py::none();
    }
}

Size2D ImageBatchVarShape::maxSize() const
{
    cv::Size2D s = m_impl.maxSize();
    return {s.w, s.h};
}

int32_t ImageBatchVarShape::capacity() const
{
    return m_impl.capacity();
}

int32_t ImageBatchVarShape::numImages() const
{
    NVCV_ASSERT(m_impl.numImages() == (int)m_list.size());
    return m_impl.numImages();
}

void ImageBatchVarShape::pushBack(Image &img)
{
    m_impl.pushBack(img.impl());
    m_list.push_back(img.shared_from_this());
}

void ImageBatchVarShape::pushBackMany(std::vector<std::shared_ptr<Image>> &imgList)
{
    // TODO: use an iterator that return the handle when dereferenced, this
    // would avoid creating this vector.
    std::vector<NVCVImageHandle> handles;
    handles.reserve(imgList.size());
    for (auto &img : imgList)
    {
        handles.push_back(img->impl().handle());
        m_list.push_back(img);
    }

    m_impl.pushBack(handles.begin(), handles.end());
}

void ImageBatchVarShape::popBack(int imgCount)
{
    m_impl.popBack(imgCount);
    m_list.erase(m_list.end() - imgCount, m_list.end());
}

void ImageBatchVarShape::clear()
{
    m_impl.clear();
    m_list.clear();
}

auto ImageBatchVarShape::begin() const -> ImageList::const_iterator
{
    return m_list.begin();
}

auto ImageBatchVarShape::end() const -> ImageList::const_iterator
{
    return m_list.end();
}

void ImageBatchVarShape::Export(py::module &m)
{
    using namespace py::literals;

    py::class_<ImageBatchVarShape, std::shared_ptr<ImageBatchVarShape>, Container>(m, "ImageBatchVarShape")
        .def(py::init(&ImageBatchVarShape::Create), "capacity"_a)
        .def_property_readonly("uniqueformat", &ImageBatchVarShape::uniqueFormat)
        .def_property_readonly("maxsize", &ImageBatchVarShape::maxSize)
        .def_property_readonly("capacity", &ImageBatchVarShape::capacity)
        .def("__len__", &ImageBatchVarShape::numImages)
        .def("__iter__", [](const ImageBatchVarShape &list) { return py::make_iterator(list); })
        .def("pushback", &ImageBatchVarShape::pushBack)
        .def("pushback", &ImageBatchVarShape::pushBackMany)
        .def("popback", &ImageBatchVarShape::popBack, "count"_a = 1)
        .def("clear", &ImageBatchVarShape::clear);
}

} // namespace nv::cvpy
