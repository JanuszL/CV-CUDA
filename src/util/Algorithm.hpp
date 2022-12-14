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

#ifndef NVCV_UTIL_ALGORITHM_HPP
#define NVCV_UTIL_ALGORITHM_HPP

namespace nv::cv::util {

template<class HEAD, class... TAIL>
constexpr auto Max(const HEAD &head, const TAIL &...tail)
{
    if constexpr (sizeof...(TAIL) == 0)
    {
        return head;
    }
    else
    {
        auto maxTail = Max(tail...);
        return head >= maxTail ? head : maxTail;
    }
}

} // namespace nv::cv::util

#endif // NVCV_UTIL_ALGORITHM_HPP
