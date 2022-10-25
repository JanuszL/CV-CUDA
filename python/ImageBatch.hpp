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

#ifndef NVCV_PYTHON_IMAGEBATCH_HPP
#define NVCV_PYTHON_IMAGEBATCH_HPP

#include "Container.hpp"

#include <nvcv/ImageBatch.hpp>

#include <vector>

namespace nv::cvpy {
namespace py = pybind11;

class Image;

class ImageBatchVarShape : public Container
{
    using ImageList = std::vector<std::shared_ptr<Image>>;

public:
    static void Export(py::module &m);

    static std::shared_ptr<ImageBatchVarShape> Create(int capacity, cv::ImageFormat fmt);

    const cv::ImageBatchVarShape &impl() const;
    cv::ImageBatchVarShape       &impl();

    // Let's simplify a bit and NOT export the base class ImageBatch,
    // as we currently have only one leaf class (this one).
    cv::ImageFormat format() const;
    int32_t         capacity() const;
    int32_t         size() const;

    void pushBack(Image &img);
    void pushBackMany(std::vector<std::shared_ptr<Image>> &imgList);
    void popBack(int imgCount);
    void clear();

    ImageList::const_iterator begin() const;
    ImageList::const_iterator end() const;

    class Key final : public IKey
    {
    public:
        Key() = default;

        explicit Key(int capacity, cv::ImageFormat fmt)
            : m_capacity(capacity)
            , m_format(fmt)
        {
        }

    private:
        int             m_capacity;
        cv::ImageFormat m_format;

        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        return m_key;
    }

private:
    explicit ImageBatchVarShape(int capacity, cv::ImageFormat fmt);
    Key                    m_key;
    ImageList              m_list;
    cv::ImageBatchVarShape m_impl;
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_IMAGEBATCH_HPP
