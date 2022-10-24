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

#include "TensorLayout.hpp"

#include "Exception.hpp"

#include <util/Assert.h>

namespace nv::cv::priv {

NVCVTensorLayout CreateLayout(const char *beg, const char *end)
{
    if (beg == nullptr || end == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Range pointers must not be NULL";
    }

    if (end - beg < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Range must not have negative length";
    }

    if (end - beg > NVCV_TENSOR_MAX_NDIM)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Range length " << end - beg << " too large, must be <= " << NVCV_TENSOR_MAX_NDIM;
    }

    NVCVTensorLayout out;
    out.ndim = end - beg;
    memcpy(out.data, beg, out.ndim);

    return out;
}

NVCVTensorLayout CreateLayout(const char *descr)
{
    NVCVTensorLayout out;

    if (descr == nullptr)
    {
        out = {};
    }
    else
    {
        const char *cur = descr;
        for (int i = 0; i < NVCV_TENSOR_MAX_NDIM && *cur; ++i, ++cur)
        {
            out.data[i] = *cur;
        }

        if (*cur != '\0')
        {
            // Avoids going through the whole descr buffer, which might pose a
            // security hazard.
            char buf[32];
            strncpy(buf, descr, 31);
            buf[31] = 0;
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Tensor layout description is too big, must have at most 16 labels: " << buf
                << (strlen(buf) <= 31 ? "" : "...");
        }

        out.ndim = cur - descr;
        NVCV_ASSERT(0 <= out.ndim && (size_t)out.ndim < sizeof(out.data) / sizeof(out.data[0]));
        out.data[out.ndim] = '\0'; // add null terminator
    }
    return out;
}

int FindDimIndex(const NVCVTensorLayout &layout, char dimLabel)
{
    if (const void *p = memchr(layout.data, dimLabel, layout.ndim))
    {
        return std::distance(reinterpret_cast<const std::byte *>(layout.data), reinterpret_cast<const std::byte *>(p));
    }
    else
    {
        return -1;
    }
}

bool IsChannelLast(const NVCVTensorLayout &layout)
{
    return layout.ndim == 0 || layout.data[layout.ndim - 1] == 'C';
}

NVCVTensorLayout CreateFirst(const NVCVTensorLayout &layout, int n)
{
    if (n >= 0)
    {
        NVCVTensorLayout out;
        out.ndim = std::min(n, layout.ndim);
        memcpy(out.data, layout.data, out.ndim);
        return out;
    }
    else
    {
        return CreateLast(layout, -n);
    }
}

NVCVTensorLayout CreateLast(const NVCVTensorLayout &layout, int n)
{
    if (n >= 0)
    {
        NVCVTensorLayout out;
        out.ndim = std::min(n, layout.ndim);
        memcpy(out.data, layout.data + layout.ndim - out.ndim, out.ndim);
        return out;
    }
    else
    {
        return CreateFirst(layout, -n);
    }
}

NVCVTensorLayout CreateSubRange(const NVCVTensorLayout &layout, int beg, int end)
{
    if (beg < 0)
    {
        beg = std::max(0, layout.ndim + beg);
    }
    else
    {
        beg = std::min(beg, layout.ndim);
    }

    if (end < 0)
    {
        end = std::max(0, layout.ndim + end);
    }
    else
    {
        end = std::min(end, layout.ndim);
    }

    NVCVTensorLayout out;

    out.ndim = end - beg;
    if (out.ndim > 0)
    {
        memcpy(out.data, layout.data + beg, out.ndim);
    }
    else
    {
        out.ndim = 0;
    }

    return out;
}

bool operator==(const NVCVTensorLayout &a, const NVCVTensorLayout &b)
{
    if (a.ndim == b.ndim)
    {
        return memcmp(a.data, b.data, a.ndim * sizeof(a.data[0])) == 0;
    }
    else
    {
        return false;
    }
}

bool operator!=(const NVCVTensorLayout &a, const NVCVTensorLayout &b)
{
    return !operator==(a, b);
}

} // namespace nv::cv::priv
