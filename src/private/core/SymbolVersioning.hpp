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

#ifndef NVCV_PRIV_CORE_SYMBOLVERSIONING_HPP
#define NVCV_PRIV_CORE_SYMBOLVERSIONING_HPP

#include <util/SymbolVersioning.hpp>

#define NVCV_DEFINE_API(...)     NVCV_PROJ_DEFINE_API(NVCV, __VA_ARGS__)
#define NVCV_DEFINE_OLD_API(...) NVCV_PROJ_DEFINE_OLD_API(NVCV, __VA_ARGS__)

#endif // NVCV_PRIV_CORE_SYMBOLVERSIONING_HPP
