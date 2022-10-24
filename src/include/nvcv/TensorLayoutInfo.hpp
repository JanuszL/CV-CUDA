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

#ifndef NVCV_TENSOR_LAYOUT_INFO_HPP
#define NVCV_TENSOR_LAYOUT_INFO_HPP

#include "TensorLayout.hpp"
#include "detail/Optional.hpp"

namespace nv { namespace cv {

class TensorLayoutInfo
{
public:
    static bool IsCompatible(const TensorLayout &layout)
    {
        return true;
    }

    static detail::Optional<TensorLayoutInfo> Create(const TensorLayout &layout)
    {
        return TensorLayoutInfo{layout};
    }

    constexpr const TensorLayout &layout() const

    {
        return m_layout;
    }

    constexpr bool isBatch() const
    {
        return m_cacheIsBatch;
    }

    int idxSample() const
    {
        return m_cacheIdxSample;
    }

    bool isImage() const
    {
        return m_cacheIsImage;
    }

protected:
    TensorLayoutInfo(const TensorLayout &layout)
        : m_layout(layout)
    {
        // isBatch ----------------
        m_cacheIsBatch = m_layout.ndim() > 0 && m_layout[0] == LABEL_BATCH;

        // isImage ----------------
        if (m_layout != TensorLayout::NONE)
        {
            m_cacheIsImage = m_layout.find(LABEL_WIDTH) >= 0;
        }
        else
        {
            m_cacheIsImage = false;
        }

        // idxSample ----------------
        m_cacheIdxSample = m_cacheIsBatch ? 0 : -1;
    }

private:
    const TensorLayout &m_layout;
    bool                m_cacheIsBatch;
    bool                m_cacheIsImage;
    int                 m_cacheIdxSample;
};

class TensorLayoutInfoImage : public TensorLayoutInfo
{
public:
    static bool IsCompatible(const TensorLayout &layout)
    {
        if (auto info = TensorLayoutInfo::Create(layout))
        {
            return info->isImage();
        }
        else
        {
            return false;
        }
    }

    static detail::Optional<TensorLayoutInfoImage> Create(const TensorLayout &layout)
    {
        if (IsCompatible(layout))
        {
            return TensorLayoutInfoImage{layout};
        }
        else
        {
            return detail::NullOpt;
        }
    }

    int numSpatialDims() const
    {
        return m_cacheNumSpatialDims;
    }

    bool isRowMajor() const
    {
        return m_cacheIsRowMajor;
    }

    // -1 if not found
    int idxChannel() const
    {
        return m_cacheIdxChannel;
    }

    // -1 if not found
    int idxWidth() const
    {
        return m_cacheIdxWidth;
    }

    // -1 if not found
    int idxHeight() const
    {
        return m_cacheIdxHeight;
    }

    // -1 if not found
    int idxDepth() const
    {
        return m_cacheIdxDepth;
    }

    bool hasChannel() const
    {
        return m_cacheHasChannel;
    }

    bool isChannelFirst() const
    {
        return m_cacheIsChannelFirst;
    }

    bool isChannelLast() const
    {
        return m_cacheIsChannelLast;
    }

protected:
    TensorLayoutInfoImage(const TensorLayout &layout)
        : TensorLayoutInfo(layout)
    {
        m_cacheNumSpatialDims = std::count_if(layout.begin(), layout.end(),
                                              [](char v)
                                              {
                                                  switch (v)
                                                  {
                                                  case LABEL_WIDTH:
                                                  case LABEL_HEIGHT:
                                                  case LABEL_DEPTH:
                                                      return true;
                                                  default:
                                                      return false;
                                                  }
                                              });

        m_cacheIsRowMajor = layout.endsWith(TensorLayout::W) || layout.endsWith(TensorLayout::WC);
        m_cacheIdxChannel = layout.find(LABEL_CHANNEL);
        m_cacheIdxWidth   = layout.find(LABEL_WIDTH);
        m_cacheIdxHeight  = layout.find(LABEL_HEIGHT);
        m_cacheIdxDepth   = layout.find(LABEL_DEPTH);
        m_cacheHasChannel = m_cacheIdxChannel >= 0;

        // isChannelFirst --------------
        if (layout != TensorLayout::NONE)
        {
            if (this->isBatch())
            {
                m_cacheIsChannelFirst = layout[1] == LABEL_CHANNEL;
            }
            else
            {
                m_cacheIsChannelFirst = layout[0] == LABEL_CHANNEL;
            }
        }
        else
        {
            m_cacheIsChannelFirst = false;
        }

        // isChannelLast --------------
        if (layout != TensorLayout::NONE)
        {
            m_cacheIsChannelLast = layout[layout.ndim() - 1] == LABEL_CHANNEL || !this->hasChannel();
        }
        else
        {
            m_cacheIsChannelLast = false;
        }
    }

private:
    int  m_cacheNumSpatialDims;
    bool m_cacheIsRowMajor;
    int  m_cacheIdxChannel;
    int  m_cacheIdxWidth;
    int  m_cacheIdxHeight;
    int  m_cacheIdxDepth;
    bool m_cacheHasChannel;
    bool m_cacheIsChannelFirst;
    bool m_cacheIsChannelLast;
};

}} // namespace nv::cv

#endif // NVCV_TENSOR_LAYOUT_INFO_HPP
