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

#ifndef NVCV_TESTS_DEVICE_SATURATE_CAST_HPP
#define NVCV_TESTS_DEVICE_SATURATE_CAST_HPP

template<typename TargetPixelType, typename SourcePixelType>
TargetPixelType DeviceRunSaturateCast(SourcePixelType);

#endif // NVCV_TESTS_DEVICE_SATURATE_CAST_HPP
