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

#ifndef NVCV_TESTS_DEVICE_BORDER_WRAP_HPP
#define NVCV_TESTS_DEVICE_BORDER_WRAP_HPP

#include <cuda_runtime.h> // for cudaStream_t, etc.

template<class DstWrapper, class SrcWrapper, typename DimType>
void DeviceRunFillBorder(DstWrapper &dstWrap, SrcWrapper &srcWrap, DimType dstSize, DimType srcSize,
                         cudaStream_t &stream);

#endif // NVCV_TESTS_DEVICE_BORDER_WRAP_HPP
