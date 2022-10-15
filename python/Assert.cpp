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

#include "Assert.hpp"

#include <cstdio>
#include <cstdlib>

namespace nv::cvpy {

void DoAssert(const char *file, int line, const char *cond)
{
#if NVCV_EXPOSE_CODE
    fprintf(stderr, "Fatal assertion error on %s:%d: %s\n", file, line, cond);
#else
    (void)file;
    (void)line;
    (void)cond;
    fprintf(stderr, "Fatal assertion error\n");
#endif
    abort();
}

} // namespace nv::cvpy
