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

#include <common/TypedTests.hpp>
#include <cvcuda/OpSIFT.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <util/TensorDataUtils.hpp>

#include <random>
#include <vector>

// ----------------------- Basic utility definitions ---------------------------

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;
namespace type = nvcv::test::type;
namespace util = nvcv::util;

using RawBufferType = std::vector<uint8_t>;

using VT = uint8_t; // value type, SIFT only accepts U8

constexpr nvcv::ImageFormat kInFormat{nvcv::FMT_U8};

static std::default_random_engine g_rng(0); // seed 0 to fix randomness

// ----------------------------- Start tests -----------------------------------

// clang-format off

#define NVCV_SHAPE(w, h, n) (int3{w, h, n})

#define NVCV_TEST_ROW(InShape, MaxFeatures, NumOctaveLayers, ContrastTh, EdgeTh, InitSigma, ExpandInput)        \
    type::Types<type::Value<InShape>, type::Value<MaxFeatures>, type::Value<NumOctaveLayers>,                   \
                type::Value<ContrastTh>, type::Value<EdgeTh>, type::Value<InitSigma>, type::Value<ExpandInput>>

NVCV_TYPED_TEST_SUITE(OpSIFT, type::Types<
    NVCV_TEST_ROW(NVCV_SHAPE(23, 17, 3), 55, 2, 0.01f, 1.f, .5f, false)
>);

// clang-format on

TYPED_TEST(OpSIFT, SIFT_sanity)
{
    int3  inShape           = type::GetValue<TypeParam, 0>;
    long  capacity          = type::GetValue<TypeParam, 1>;
    int   numOctaveLayers   = type::GetValue<TypeParam, 2>;
    float contrastThreshold = type::GetValue<TypeParam, 3>;
    float edgeThreshold     = type::GetValue<TypeParam, 4>;
    float initSigma         = type::GetValue<TypeParam, 5>;
    bool  expandInput       = type::GetValue<TypeParam, 6>;

    NVCVSIFTFlagType flags = expandInput ? NVCV_SIFT_USE_EXPANDED_INPUT : NVCV_SIFT_USE_ORIGINAL_INPUT;

    // Increasing inShape and numOctaveLayers to test bigger maxShape and maxOctaveLayers
    int3 maxShape        = (inShape + 3) * (expandInput ? 2 : 1);
    int  maxOctaveLayers = numOctaveLayers + 1;

    nvcv::Tensor src = nvcv::util::CreateTensor(inShape.z, inShape.x, inShape.y, kInFormat);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    long3 srcShape{srcAccess->numSamples(), srcAccess->numRows(), srcAccess->numCols()};
    long3 srcStrides{srcAccess->sampleStride(), srcAccess->rowStride(), srcAccess->colStride()};

    // While inShape is WHN, srcShape is NHW to match srcStrides
    ASSERT_TRUE(inShape.z == srcShape.x && inShape.y == srcShape.y && inShape.x == srcShape.z);

    srcStrides.x = (srcData->rank() == 3) ? srcShape.y * srcStrides.y : srcStrides.x;

    long srcBufSize = srcStrides.x * srcShape.x;

    RawBufferType srcVec(srcBufSize);

    // clang-format off

    std::uniform_int_distribution<VT> rg(0, 255);

    for (long x = 0; x < srcShape.x; ++x)
        for (long y = 0; y < srcShape.y; ++y)
            for (long z = 0; z < srcShape.z; ++z)
                util::ValueAt<VT>(srcVec, srcStrides, long3{x, y, z}) = rg(g_rng);

    nvcv::Tensor featCoords({{srcShape.x, capacity}, "NM"}, nvcv::TYPE_4F32);
    nvcv::Tensor featMetadata({{srcShape.x, capacity}, "NM"}, nvcv::TYPE_3F32);
    nvcv::Tensor featDescriptors({{srcShape.x, capacity}, "NM"}, nvcv::TYPE_2U64);
    nvcv::Tensor numFeatures({{srcShape.x}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), srcBufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::SIFT op(maxShape, maxOctaveLayers);

    EXPECT_NO_THROW(op(stream, src, featCoords, featMetadata, featDescriptors, numFeatures, numOctaveLayers,
                       contrastThreshold, edgeThreshold, initSigma, flags));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
