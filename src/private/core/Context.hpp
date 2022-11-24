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

#ifndef NVCV_PRIV_CORE_CONTEXT_HPP
#define NVCV_PRIV_CORE_CONTEXT_HPP

#include "DefaultAllocator.hpp"
#include "IContext.hpp"

namespace nv::cv::priv {

class Context final : public IContext
{
public:
    Context();
    ~Context();

    IAllocator &allocDefault() override;

private:
    DefaultAllocator m_allocDefault;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_CONTEXT_HPP
