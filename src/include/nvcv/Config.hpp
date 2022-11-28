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

/**
 * @file Config.hpp
 *
 * @brief Public C++ interface to NVCV configuration.
 */

#ifndef NVCV_CONFIG_HPP
#define NVCV_CONFIG_HPP

#include "Config.h"
#include "detail/CheckError.hpp"

namespace nv { namespace cv { namespace cfg {

inline void SetMaxImageCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxImageCount(maxCount));
}

inline void SetMaxImageBatchCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxImageBatchCount(maxCount));
}

inline void SetMaxTensorCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxTensorCount(maxCount));
}

inline void SetMaxAllocatorCount(int32_t maxCount)
{
    detail::CheckThrow(nvcvConfigSetMaxAllocatorCount(maxCount));
}

}}} // namespace nv::cv::cfg

#endif // NVCV_CONFIG_HPP
