/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_TENSORDATAACESSOR_HPP
#define NVCV_TENSORDATAACESSOR_HPP

#include "ITensorData.hpp"
#include "TensorShapeInfo.hpp"
#include "detail/BaseFromMember.hpp"

#include <cstddef>

namespace nv { namespace cv {

// Design is similar to TensorShapeInfo hierarchy

namespace detail {

class TensorDataAccessPitchImpl
{
public:
    TensorDataAccessPitchImpl(const ITensorDataPitch &tdata, const TensorShapeInfoImpl &infoShape)
        : m_tdata(tdata)
        , m_infoShape(infoShape)
    {
    }

    TensorShape::DimType numSamples() const
    {
        return m_infoShape.numSamples();
    }

    DataType dtype() const
    {
        return m_tdata.dtype();
    }

    const TensorLayout &layout() const
    {
        return m_tdata.layout();
    }

    const TensorShape &shape() const
    {
        return m_tdata.shape();
    }

    int64_t samplePitchBytes() const
    {
        int idx = this->infoLayout().idxSample();
        if (idx >= 0)
        {
            return m_tdata.pitchBytes(idx);
        }
        else
        {
            return 0;
        }
    }

    void *sampleData(int n) const
    {
        return sampleData(n, m_tdata.data());
    }

    void *sampleData(int n, void *base) const
    {
        assert(0 <= n && n < this->numSamples());
        return reinterpret_cast<std::byte *>(base) + this->samplePitchBytes() * n;
    }

    bool isImage() const
    {
        return m_infoShape.isImage();
    }

    const TensorShapeInfoImpl &infoShape() const
    {
        return m_infoShape;
    }

    const TensorLayoutInfo &infoLayout() const
    {
        return m_infoShape.infoLayout();
    }

protected:
    const ITensorDataPitch &m_tdata;

    TensorDataAccessPitchImpl(const TensorDataAccessPitchImpl &that) = delete;

    TensorDataAccessPitchImpl(const TensorDataAccessPitchImpl &that, const TensorShapeInfoImpl &infoShape)
        : m_tdata(that.m_tdata)
        , m_infoShape(infoShape)
    {
    }

private:
    const TensorShapeInfoImpl &m_infoShape;
};

class TensorDataAccessPitchImageImpl : public TensorDataAccessPitchImpl
{
public:
    TensorDataAccessPitchImageImpl(const ITensorDataPitch &tdata, const TensorShapeInfoImageImpl &infoShape)
        : TensorDataAccessPitchImpl(tdata, infoShape)
    {
    }

    const TensorShapeInfoImageImpl &infoShape() const
    {
        return static_cast<const TensorShapeInfoImageImpl &>(TensorDataAccessPitchImpl::infoShape());
    }

    const TensorLayoutInfoImage &infoLayout() const
    {
        return this->infoShape().infoLayout();
    }

    int32_t numCols() const
    {
        return this->infoShape().numCols();
    }

    int32_t numRows() const
    {
        return this->infoShape().numRows();
    }

    int32_t numChannels() const
    {
        return this->infoShape().numChannels();
    }

    Size2D size() const
    {
        return this->infoShape().size();
    }

    int64_t chPitchBytes() const
    {
        int idx = this->infoLayout().idxChannel();
        if (idx >= 0)
        {
            return m_tdata.pitchBytes(idx);
        }
        else
        {
            return 0;
        }
    }

    int64_t colPitchBytes() const
    {
        int idx = this->infoLayout().idxWidth();
        if (idx >= 0)
        {
            return m_tdata.pitchBytes(idx);
        }
        else
        {
            return 0;
        }
    }

    int64_t rowPitchBytes() const
    {
        int idx = this->infoLayout().idxHeight();
        if (idx >= 0)
        {
            return m_tdata.pitchBytes(idx);
        }
        else
        {
            return 0;
        }
    }

    int64_t depthPitchBytes() const
    {
        int idx = this->infoLayout().idxDepth();
        if (idx >= 0)
        {
            return m_tdata.pitchBytes(idx);
        }
        else
        {
            return 0;
        }
    }

    void *rowData(int y) const
    {
        return rowData(y, m_tdata.data());
    }

    void *rowData(int y, void *base) const
    {
        assert(0 <= y && y < this->numRows());
        return reinterpret_cast<std::byte *>(base) + this->rowPitchBytes() * y;
    }

    void *chData(int c) const
    {
        return chData(c, m_tdata.data());
    }

    void *chData(int c, void *base) const
    {
        assert(0 <= c && c < this->numChannels());
        return reinterpret_cast<std::byte *>(base) + this->chPitchBytes() * c;
    }

protected:
    TensorDataAccessPitchImageImpl(const TensorDataAccessPitchImageImpl &that) = delete;

    TensorDataAccessPitchImageImpl(const TensorDataAccessPitchImageImpl &that,
                                   const TensorShapeInfoImageImpl       &infoShape)
        : TensorDataAccessPitchImpl(that, infoShape)
    {
    }
};

class TensorDataAccessPitchImagePlanarImpl : public TensorDataAccessPitchImageImpl
{
public:
    TensorDataAccessPitchImagePlanarImpl(const ITensorDataPitch &tdata, const TensorShapeInfoImagePlanar &infoShape)
        : TensorDataAccessPitchImageImpl(tdata, infoShape)
    {
    }

    const TensorShapeInfoImagePlanar &infoShape() const
    {
        return static_cast<const TensorShapeInfoImagePlanar &>(TensorDataAccessPitchImageImpl::infoShape());
    }

    int32_t numPlanes() const
    {
        return this->infoShape().numPlanes();
    }

    int64_t planePitchBytes() const
    {
        if (this->infoLayout().isChannelFirst())
        {
            int ichannel = this->infoLayout().idxChannel();
            assert(ichannel >= 0);
            return m_tdata.pitchBytes(ichannel);
        }
        else
        {
            return 0;
        }
    }

    void *planeData(int p) const
    {
        return planeData(p, m_tdata.data());
    }

    void *planeData(int p, void *base) const
    {
        assert(0 <= p && p < this->numPlanes());
        return reinterpret_cast<std::byte *>(base) + this->planePitchBytes() * p;
    }

protected:
    TensorDataAccessPitchImagePlanarImpl(const TensorDataAccessPitchImagePlanarImpl &that) = delete;

    TensorDataAccessPitchImagePlanarImpl(const TensorDataAccessPitchImagePlanarImpl &that,
                                         const TensorShapeInfoImagePlanar           &infoShape)
        : TensorDataAccessPitchImageImpl(that, infoShape)
    {
    }
};

} // namespace detail

class TensorDataAccessPitch
    // declaration order is important here
    : private detail::BaseFromMember<TensorShapeInfo>
    , public detail::TensorDataAccessPitchImpl
{
public:
    static bool IsCompatible(const ITensorData &data)
    {
        return dynamic_cast<const ITensorDataPitch *>(&data) != nullptr;
    }

    static detail::Optional<TensorDataAccessPitch> Create(const ITensorData &data)
    {
        if (auto *dataPitch = dynamic_cast<const ITensorDataPitch *>(&data))
        {
            return TensorDataAccessPitch(*dataPitch);
        }
        else
        {
            return detail::NullOpt;
        }
    }

    TensorDataAccessPitch(const TensorDataAccessPitch &that)
        : MemberShapeInfo(that)
        , detail::TensorDataAccessPitchImpl(that, MemberShapeInfo::member)
    {
    }

private:
    using MemberShapeInfo = detail::BaseFromMember<TensorShapeInfo>;

    TensorDataAccessPitch(const ITensorDataPitch &data)
        : MemberShapeInfo{*TensorShapeInfo::Create(data.shape())}
        , detail::TensorDataAccessPitchImpl(data, MemberShapeInfo::member)
    {
    }
};

class TensorDataAccessPitchImage
    // declaration order is important here
    : private detail::BaseFromMember<TensorShapeInfoImage>
    , public detail::TensorDataAccessPitchImageImpl
{
public:
    TensorDataAccessPitchImage(const TensorDataAccessPitchImage &that)
        : MemberShapeInfo(that)
        , detail::TensorDataAccessPitchImageImpl(that, MemberShapeInfo::member)
    {
    }

    static bool IsCompatible(const ITensorData &data)
    {
        return TensorDataAccessPitch::IsCompatible(data) && TensorShapeInfoImage::IsCompatible(data.shape());
    }

    static detail::Optional<TensorDataAccessPitchImage> Create(const ITensorData &data)
    {
        if (IsCompatible(data))
        {
            return TensorDataAccessPitchImage(dynamic_cast<const ITensorDataPitch &>(data));
        }
        else
        {
            return detail::NullOpt;
        }
    }

private:
    using MemberShapeInfo = detail::BaseFromMember<TensorShapeInfoImage>;

protected:
    TensorDataAccessPitchImage(const ITensorDataPitch &data)
        : MemberShapeInfo{*TensorShapeInfoImage::Create(data.shape())}
        , detail::TensorDataAccessPitchImageImpl(data, MemberShapeInfo::member)
    {
    }
};

class TensorDataAccessPitchImagePlanar
    // declaration order is important here
    : private detail::BaseFromMember<TensorShapeInfoImagePlanar>
    , public detail::TensorDataAccessPitchImagePlanarImpl
{
public:
    TensorDataAccessPitchImagePlanar(const TensorDataAccessPitchImagePlanar &that)
        : MemberShapeInfo(that)
        , detail::TensorDataAccessPitchImagePlanarImpl(that, MemberShapeInfo::member)
    {
    }

    static bool IsCompatible(const ITensorData &data)
    {
        return TensorDataAccessPitchImage::IsCompatible(data) && TensorShapeInfoImagePlanar::IsCompatible(data.shape());
    }

    static detail::Optional<TensorDataAccessPitchImagePlanar> Create(const ITensorData &data)
    {
        if (IsCompatible(data))
        {
            return TensorDataAccessPitchImagePlanar(dynamic_cast<const ITensorDataPitch &>(data));
        }
        else
        {
            return detail::NullOpt;
        }
    }

private:
    using MemberShapeInfo = detail::BaseFromMember<TensorShapeInfoImagePlanar>;

protected:
    TensorDataAccessPitchImagePlanar(const ITensorDataPitch &data)
        : MemberShapeInfo{*TensorShapeInfoImagePlanar::Create(data.shape())}
        , detail::TensorDataAccessPitchImagePlanarImpl(data, MemberShapeInfo::member)
    {
    }
};

}} // namespace nv::cv

#endif // NVCV_TENSORDATAACESSOR_HPP
