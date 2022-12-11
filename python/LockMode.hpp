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

#ifndef NVCV_PYTHON_LOCKMODE_HPP
#define NVCV_PYTHON_LOCKMODE_HPP

namespace nv::cvpy {

enum LockMode
{
    LOCK_NONE      = 0,
    LOCK_READ      = 1,
    LOCK_WRITE     = 2,
    LOCK_READWRITE = LOCK_READ | LOCK_WRITE
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_LOCKMODE_HPP
