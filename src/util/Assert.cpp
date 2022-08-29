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

#include "Assert.h"

#include <cstdio>
#include <cstdlib>

extern "C" void NvCVAssert(const char *file, int line, const char *cond)
{
    if (file == nullptr)
    {
        fprintf(stderr, "Fatal assertion error\n");
    }
    else
    {
        fprintf(stderr, "Fatal assertion error on %s:%d: (%s) failed\n", file, line, cond);
    }

    abort();
}
