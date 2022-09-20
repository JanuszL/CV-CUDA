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

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <util/CheckError.hpp>
#include <util/Exception.hpp>

namespace gt   = ::testing;
namespace test = nv::cv::test;

// clang-format off
NVCV_TEST_SUITE_P(CheckErrorCudaConversionTests, test::ValueList<cudaError_t, NVCVStatus>
{
     { cudaErrorMemoryAllocation,   NVCV_ERROR_OUT_OF_MEMORY    },
     { cudaErrorNotReady,           NVCV_ERROR_NOT_READY        },
     { cudaErrorInvalidValue,       NVCV_ERROR_INVALID_ARGUMENT },
     { cudaErrorTextureFetchFailed, NVCV_ERROR_INTERNAL         }
});

// clang-format on

TEST_P(CheckErrorCudaConversionTests, check_conversion_to_nvcvstatus)
{
    cudaError_t errCuda = GetParamValue<0>();
    NVCVStatus  gold    = GetParamValue<1>();

    NVCV_EXPECT_STATUS(gold, NVCV_CHECK_THROW(errCuda));
}

TEST(CheckErrorCudaTests, success_no_throw)
{
    EXPECT_NO_THROW(NVCV_CHECK_THROW(cudaSuccess));
}
