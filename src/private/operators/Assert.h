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

#ifndef NVCV_OP_PRIV_ASSERT_H
#define NVCV_OP_PRIV_ASSERT_H

#ifdef __cplusplus
extern "C"
{
#endif

#if !defined(NVCV_CUDA_ASSERT)
#    if !defined(NDEBUG) && __NVCC__
#        define NVCV_CUDA_ASSERT(x, ...)                     \
            do                                               \
            {                                                \
                if (!(x))                                    \
                {                                            \
                    printf("E Condition (%s) failed: ", #x); \
                    printf(__VA_ARGS__);                     \
                    asm("trap;");                            \
                }                                            \
            }                                                \
            while (1 == 0)
#    else
#        define NVCV_CUDA_ASSERT(x, ...) \
            do                           \
            {                            \
            }                            \
            while (1 == 0)
#    endif
#endif // !defined(NVCV_CUDA_ASSERT)

#ifdef __cplusplus
}
#endif

#endif // NVCV_OP_PRIV_ASSERT_H
