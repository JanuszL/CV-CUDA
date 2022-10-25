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

#ifndef NVCV_PYTHON_IMAGE_HPP
#define NVCV_PYTHON_IMAGE_HPP

#include "Container.hpp"
#include "CudaBuffer.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"

#include <nvcv/Image.hpp>
#include <nvcv/ImageFormat.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <variant>

namespace nv::cvpy {
namespace py = pybind11;

class Image final : public Container
{
public:
    static void Export(py::module &m);

    static std::shared_ptr<Image> Zeros(const Size2D &size, cv::ImageFormat fmt);
    static std::shared_ptr<Image> Create(const Size2D &size, cv::ImageFormat fmt);
    static std::shared_ptr<Image> CreateHost(py::buffer buffer, cv::ImageFormat fmt);
    static std::shared_ptr<Image> CreateHostVector(std::vector<py::buffer> buffer, cv::ImageFormat fmt);

    static std::shared_ptr<Image> WrapDevice(CudaBuffer &buffer, cv::ImageFormat fmt);
    static std::shared_ptr<Image> WrapDeviceVector(std::vector<std::shared_ptr<CudaBuffer>> buffer,
                                                   cv::ImageFormat                          fmt);

    std::shared_ptr<Image>       shared_from_this();
    std::shared_ptr<const Image> shared_from_this() const;

    Size2D          size() const;
    int32_t         width() const;
    int32_t         height() const;
    cv::ImageFormat format() const;

    friend std::ostream &operator<<(std::ostream &out, const Image &img);

    cv::IImage &impl()
    {
        return *m_impl;
    }

    const cv::IImage &impl() const
    {
        return *m_impl;
    }

    class Key final : public IKey
    {
    public:
        Key() = default;

        explicit Key(Size2D size, cv::ImageFormat fmt)
            : m_size(size)
            , m_format(fmt)
        {
        }

    private:
        Size2D          m_size;
        cv::ImageFormat m_format;

        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        return m_key;
    }

private:
    explicit Image(const Size2D &size, cv::ImageFormat fmt);
    explicit Image(std::vector<std::shared_ptr<CudaBuffer>> buf, const cv::IImageDataPitchDevice &imgData);
    explicit Image(std::vector<py::buffer> buf, const cv::IImageDataPitchHost &imgData);

    std::unique_ptr<cv::IImage> m_impl; // must come before m_key
    Key                         m_key;

    // monostate (empty) if not wrapping
    // It could be wrapping either host (py::buffer) or cuda (CudaBuffer)
    std::variant<std::monostate, std::vector<py::buffer>, std::vector<std::shared_ptr<CudaBuffer>>> m_wrapped;
};

std::ostream &operator<<(std::ostream &out, const Image &img);

} // namespace nv::cvpy

#endif // NVCV_PYTHON_IMAGE_HPP
