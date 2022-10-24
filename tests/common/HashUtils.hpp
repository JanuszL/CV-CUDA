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

#ifndef NVCV_TESTS_COMMON_HASHUTILS_HPP
#define NVCV_TESTS_COMMON_HASHUTILS_HPP

#include <nvcv/TensorShape.hpp>
#include <util/HashMD5.hpp>

namespace nv::cv {

inline void Update(util::HashMD5 &hash, const TensorShape &ts)
{
    Update(hash, ts.shape(), ts.layout());
}

} // namespace nv::cv

#endif // NVCV_TESTS_COMMON_HASHUTILS_HPP
