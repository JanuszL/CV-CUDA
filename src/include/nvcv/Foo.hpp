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

#ifndef NVCV_FOO_HPP
#define NVCV_FOO_HPP

#include "Export.h"

/**
* @file Foo.hpp
*
* @brief Foo : Interface for all the image processing utilities.
*
*/

namespace nv::cv {

/**
* @brief Test function
*
* @param[in] value Input test values
*
* @returns bool value
*/
NVCV_PUBLIC bool Foo(int value);

} // namespace nv::cv

#endif // NVCV_FOO_HPP
