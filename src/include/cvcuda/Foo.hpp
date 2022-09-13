/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NV_CVCUDA_FOO_HPP
#define NV_CVCUDA_FOO_HPP

#include "Export.h"

/**
* @file Foo.hpp
*
* @brief Foo : Interface for all the image processing utilities.
*
*/

namespace nv::cuda {

/**
* @brief Test function
*
* @param[in] value Input test values
*
* @returns bool value
*/
CVCUDA_PUBLIC bool Foo(int value);

} // namespace nv::cuda

#endif // NV_CVCUDA_FOO_HPP
