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

__asm__(".symver weak_ver1_data,weak_data@ver_1.0");
__attribute__((weak)) void *weak_ver1_data = (void *)0;
;

__asm__(".symver weak_ver2_data,weak_data@@ver_2.0");
__attribute__((weak)) void *weak_ver2_data = (void *)0;

__asm__(".symver strong_ver1_data,strong_data@ver_1.0");
void *strong_ver1_data = (void *)0;
;

__asm__(".symver strong_ver2_data,strong_data@@ver_2.0");
void *strong_ver2_data = (void *)0;
;

__asm__(".symver weak_ver1_func,weak_func@ver_1.0");

__attribute__((weak)) void weak_ver1_func() {}

__asm__(".symver weak_ver2_func,weak_func@@ver_2.0");

__attribute__((weak)) void weak_ver2_func() {}

__asm__(".symver strong_ver1_func,strong_func@ver_1.0");

void strong_ver1_func() {}

__asm__(".symver strong_ver2_func,strong_func@@ver_2.0");

void strong_ver2_func() {}

__asm__(".symver clash_ver2_func,clash_func_@@ver_2.0");

void clash_ver2_func() {}

void clash_func() {}
