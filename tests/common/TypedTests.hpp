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

#ifndef NVCV_TEST_COMMON_TYPEDTESTS_HPP
#define NVCV_TEST_COMMON_TYPEDTESTS_HPP

#include "TypeList.hpp"

#include <gtest/gtest.h>

// Helper to be able to define typed test cases by defining the
// types inline (no worries about commas in macros)

#define NVCV_TYPED_TEST_SUITE_F(TEST, ...) \
    using TEST##_Types = __VA_ARGS__;      \
    TYPED_TEST_SUITE(TEST, TEST##_Types)

#define NVCV_TYPED_TEST_SUITE(TEST, ...) \
    template<class T>                    \
    class TEST : public ::testing::Test  \
    {                                    \
    };                                   \
    NVCV_TYPED_TEST_SUITE_F(TEST, __VA_ARGS__)

#define NVCV_INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, ...) \
    using TEST##INSTNAME##_Types = __VA_ARGS__;                  \
    INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, TEST##INSTNAME##_Types)

#endif // NVCV_TEST_COMMON_TYPEDTESTS_HPP
