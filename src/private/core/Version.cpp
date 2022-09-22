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

#include "Version.hpp"

#include <iostream>

namespace nv::cv::priv {

std::ostream &operator<<(std::ostream &out, const Version &ver)
{
    out << 'v' << ver.major() << '.' << ver.minor() << '.' << ver.patch();
    if (ver.tweak() != 0)
    {
        out << '.' << ver.tweak();
    }
    return out;
}

} // namespace nv::cv::priv
