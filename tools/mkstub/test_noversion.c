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

__attribute__((weak)) void *weak_nv_data = (void *)0;

void *strong_nv_data = (void *)0;

void *strong_common_data;

__attribute__((weak)) void weak_nv_func() {}

void strong_nv_func() {}

__attribute__((weak)) extern void *weak_undef_data;

extern void *strong_undef_data;

__attribute__((weak)) extern void weak_undef_func();

extern void strong_undef_func();

__thread void *tls_strong_nv_data = (void *)0;

static void ifunc_aux() {}

static void (*ifunc_resolver())()
{
    return ifunc_aux;
}

/* Define it before strong_ifunc on purpose, weak ifuncs must be
 * emitted after all strong ifuncs are emitted
 * gcc doesn't allow us to define directly a weak indirect function, we must do
 * it via a weak alias to a strong ifunc
 */
extern void weak_ifunc() __attribute__((weak, alias("strong_ifunc")));

extern void strong_ifunc() __attribute__((ifunc("ifunc_resolver")));
