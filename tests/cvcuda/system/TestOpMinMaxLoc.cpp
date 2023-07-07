/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpMinMaxLoc.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

// clang-format off

NVCV_TEST_SUITE_P(OpMinMaxLoc, test::ValueList<int, int, int, NVCVImageFormat, int, NVCVDataType>
{
    // inBatch, inWidth, inHeight,                format, maxNumLocations,       valDataType,
    {        1,       2,        2,  NVCV_IMAGE_FORMAT_U8,              10, NVCV_DATA_TYPE_U32, },
});

// clang-format on

TEST_P(OpMinMaxLoc, sanity)
{
    int batches = GetParamValue<0>();
    int width   = GetParamValue<1>();
    int height  = GetParamValue<2>();

    int3 inShape{width, height, batches};

    nvcv::ImageFormat format{GetParamValue<3>()};

    int capacity = GetParamValue<4>();

    nvcv::DataType valDataType{GetParamValue<5>()};

    nvcv::Tensor in = nvcv::util::CreateTensor(batches, width, height, format);

    // clang-format off

    nvcv::Tensor minVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor minLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{inShape.z}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor maxVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor maxLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{inShape.z}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinMaxLoc op;
    EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpMinMaxLoc, varshape_sanity)
{
    int batches = GetParamValue<0>();
    int width   = GetParamValue<1>();
    int height  = GetParamValue<2>();

    int3 inShape{width, height, batches};

    nvcv::ImageFormat format{GetParamValue<3>()};

    int capacity = GetParamValue<4>();

    nvcv::DataType valDataType{GetParamValue<5>()};

    std::vector<nvcv::Image> inImg;

    for (int z = 0; z < batches; ++z)
    {
        inImg.emplace_back(nvcv::Size2D{width, height}, format);
    }

    nvcv::ImageBatchVarShape in(batches);
    in.pushBack(inImg.begin(), inImg.end());

    // clang-format off

    nvcv::Tensor minVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor minLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{inShape.z}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor maxVal({{inShape.z}, "N"}, valDataType);
    nvcv::Tensor maxLoc({{inShape.z, capacity}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{inShape.z}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinMaxLoc op;
    EXPECT_NO_THROW(op(stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
