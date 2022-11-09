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

#ifndef NVCV_UTIL_EXCEPTION_HPP
#define NVCV_UTIL_EXCEPTION_HPP

#if NVCV_EXPORTING
#    include <private/core/Exception.hpp>
#else
#    include <nvcv/Exception.hpp>
#endif

namespace nv::cv::util {

#if NVCV_EXPORTING
using cv::priv::Exception;
#else
using cv::Exception;
#endif

} // namespace nv::cv::util

#endif // NVCV_UTIL_EXCEPTION_HPP
