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

namespace gt   = ::testing;
namespace test = nv::cv::test;
namespace priv = nv::cv::priv;

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

NVCV_TEST_SUITE_P(CheckStatusMacroTests, test::ValueList{NVCV_SUCCESS, NVCV_ERROR_NOT_READY, NVCV_ERROR_INTERNAL});

TEST_P(CheckStatusMacroTests, return_value)
{
    const NVCVStatus status = GetParam();

    int a = 0; // so that we have a colon in the macro

    NVCV_EXPECT_STATUS(status, [a, status] { return status; })
    NVCV_ASSERT_STATUS(status, [a, status] { return status; })
}

TEST_P(CheckStatusMacroTests, throw_return_void)
{
    const NVCVStatus status = GetParam();

    int a = 0; // so that we have a colon in the macro

    NVCV_EXPECT_STATUS(status, [a, status] { throw priv::Exception(status, "."); })
    NVCV_ASSERT_STATUS(status, [a, status] { throw priv::Exception(status, "."); })
}

TEST_P(CheckStatusMacroTests, throw_return_something_else)
{
    const NVCVStatus status = GetParam();

    int a = 0; // so that we have a colon in the macro

    NVCV_EXPECT_STATUS(status,
                       [a, status]
                       {
                           throw priv::Exception(status, ".");
                           return a;
                       })
    NVCV_ASSERT_STATUS(status,
                       [a, status]
                       {
                           throw priv::Exception(status, ".");
                           return a;
                       })
}
