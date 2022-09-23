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

#include "String.hpp"

#include "Assert.h"

#include <cstring>

namespace nv::cv::util {

void ReplaceAllInline(char *strBuffer, int bufferSize, std::string_view what, std::string_view replace) noexcept
{
    NVCV_ASSERT(strBuffer != nullptr);
    NVCV_ASSERT(bufferSize >= 1);

    std::string_view str(strBuffer);

    std::string_view::size_type pos;
    while ((pos = str.find(what, 0)) != std::string_view::npos)
    {
        // First create some space to write 'replace'.

        // Number of bytes to move
        int count_orig = str.size() - pos - what.size();
        // Make sure we won't write past end of buffer
        int count = std::min<int>(pos + replace.size() + count_orig, bufferSize) - replace.size() - pos;
        NVCV_ASSERT(count >= 0);
        // Since buffers might overlap, let's use memmove
        memmove(strBuffer + pos + replace.size(), strBuffer + pos + what.size(), count);

        // Now copy the new string, replacing 'what'
        replace.copy(strBuffer + pos, replace.size());

        // Let's set strBuffer/set to where next search must start so that we don't search into the
        // replaced string.
        strBuffer += pos + replace.size();
        str = std::string_view(strBuffer, count);

        strBuffer[str.size()] = '\0'; // make sure strBuffer is zero-terminated
    }
}

} // namespace nv::cv::util
