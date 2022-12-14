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
