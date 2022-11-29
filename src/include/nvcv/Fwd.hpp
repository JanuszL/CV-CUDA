
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

/**
 * @file Fwd.hpp
 *
 * @brief Forward declaration of some public C++ interface entities.
 */

#ifndef NVCV_FWD_HPP
#define NVCV_FWD_HPP

#include "alloc/Fwd.hpp"

namespace nv::cv {

class IImage;
class IImageData;
class IImageDataCudaArray;
class IImageDataPitch;
class IImageDataPitchDevice;
class IImageDataPitchHost;

class IImageBatch;
class IImageBatchData;

class IImageBatchVarShape;
class IImageBatchVarShapeData;
class IImageBatchVarShapeDataPitch;
class IImageBatchVarShapeDataPitchDevice;

class ITensor;
class ITensorData;
class ITensorDataPitch;
class ITensorDataPitchDevice;

} // namespace nv::cv

#endif // NVCV_FWD_HPP
