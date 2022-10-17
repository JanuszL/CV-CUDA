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

#ifndef NVCV_TESTS_DEVICE_MATH_WRAPPERS_HPP
#define NVCV_TESTS_DEVICE_MATH_WRAPPERS_HPP

template<typename Type>
Type DeviceRunRoundSameType(Type);

template<typename TargetType, typename SourceType>
TargetType DeviceRunRoundDiffType(SourceType);

template<typename Type>
Type DeviceRunMin(Type, Type);

template<typename Type>
Type DeviceRunMax(Type, Type);

template<typename Type>
Type DeviceRunExp(Type);

template<typename Type>
Type DeviceRunSqrt(Type);

template<typename Type>
Type DeviceRunAbs(Type);

#endif // NVCV_TESTS_DEVICE_MATH_WRAPPERS_HPP
