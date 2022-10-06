/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_UTIL_SYMBOLVERSIONING_HPP
#define NVCV_UTIL_SYMBOLVERSIONING_HPP

#include <nvcv/detail/Export.h>

/* Tools to help defining versioned APIs.
 *
At first, all public functions are defined like this:

Header:
 NVCV_PUBLIC NVCVStatus nvcvFooCreate(uint64_t flags, NVCVFoo *handle);

Implementation:
 NVCV_DEFINE_API(1, 0, NVCVStatus, nvcvFooCreate, (uint64_t flags, NVCVFoo *handle))
 {
    implementation 1;
 }

If we need to add a new version of nvcvFooCreate, we update cv-cuda code like this:

Header:
 #if NVCV_API_VERSION == 100
 __asm__(".symver nvcvFooCreate,nvcvFooCreate@NVCV_1.0");
 NVCV_PUBLIC NVCVStatus nvcvFooCreate(uint64_t flags, NVCVFoo *handle);
 #else
 NVCV_PUBLIC NVCVStatus nvcvFooCreate(int32_t size, uint64_t flags, NVCVFoo *handle);
 #endif

Implementation:

 NVCV_DEFINE_API_OLD(1, 0, NVCVStatus, nvcvFooCreate(uint64_t flags, NVCVFoo *handle))
 {
    implementation 1;
 }

 NVCV_DEFINE_API(1, 1, NVCVStatus, nvcvFooCreate, (int32_t size, uint64_t flags, NVCVFoo *handle))
 {
    implementation 2;
 }

If user defines NVCV_API_VERSION == 100, he will use the first definition and linker will
link to the cvcuda-1.0 API. If nothing is defined, he will use the most recent definition.

Users using dlopen to retrieve our functions will have to use dlvsym to specify
what version to get. Regular dlsym will always pick the most recent one.

void *foo10 = dlvsym(lib, "nvcvFoo", "NVCV_1.0");
void *foo11 = dlvsym(lib, "nvcvFoo", "NVCV_1.1");
void *foo11_tmp = dlsym(lib, "nvcvFoo");
assert(foo11 == foo11_tmp);
*/

#define NVCV_FUNCTION_API(VER_MAJOR, VER_MINOR, NAME) NAME##_v##VER_MAJOR##_##VER_MINOR

#define NVCV_DEFINE_API_HELPER(VER_MAJOR, VER_MINOR, VERTYPE, RETTYPE, NAME, ARGS)                              \
    extern "C" __attribute__((__symver__(#NAME VERTYPE "NVCV_" #VER_MAJOR "." #VER_MINOR))) NVCV_PUBLIC RETTYPE \
        NVCV_FUNCTION_API(VER_MAJOR, VER_MINOR, NAME) ARGS

#define NVCV_DEFINE_API_OLD(VER_MAJOR, VER_MINOR, RETTYPE, NAME, ARGS) \
    NVCV_DEFINE_API_HELPER(VER_MAJOR, VER_MINOR, "@", RETTYPE, NAME, ARGS)

#define NVCV_DEFINE_API(VER_MAJOR, VER_MINOR, RETTYPE, NAME, ARGS) \
    NVCV_DEFINE_API_HELPER(VER_MAJOR, VER_MINOR, "@@", RETTYPE, NAME, ARGS)

#endif // NVCV_UTIL_SYMBOLVERSIONING_HPP
