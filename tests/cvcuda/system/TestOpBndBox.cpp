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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpBndBox.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

// static void setGoldBuffer(std::vector<uint8_t> &vect, const nvcv::TensorDataAccessStridedImagePlanar &data,
//                           NVCVRectI region, uint8_t val)
// {
//     int bytesPerChan  = data.dtype().bitsPerChannel()[0] / 8;
//     int bytesPerPixel = data.numChannels() * bytesPerChan;

//     uint8_t *ptrTop = vect.data();
//     for (int img = 0; img < data.numSamples(); img++)
//     {
//         uint8_t *ptr = ptrTop + data.sampleStride() * img;
//         for (int i = 0; i < region.height; i++)
//         {
//             memset(ptr, val, region.width * bytesPerPixel);
//             ptr += data.rowStride();
//         }
//     }
// }

// clang-format off
NVCV_TEST_SUITE_P(OpBndBox, test::ValueList<int, int, int, int, int, int, int, bool>
{
    // inWidth, inHeight, bboxX, bboxY, bboxW, bboxH, thickness,  MSAA
    {      4,      4,    0,    0,    2,    2,         -1, true },
    {      4,      4,    0,    0,    2,    2,         -1, false },

});

// clang-format on
TEST_P(OpBndBox, BndBox_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int     inWidth        = GetParamValue<0>();
    int     inHeight       = GetParamValue<1>();
    int     bboxX          = GetParamValue<2>();
    int     bboxY          = GetParamValue<3>();
    int     bboxW          = GetParamValue<4>();
    int     bboxH          = GetParamValue<5>();
    int     thickness      = GetParamValue<6>();
    bool    MSAA           = GetParamValue<7>();

    uchar4  borderColor    = { 255, 0, 0, 255};
    uchar4  fillColor      = { 0, 0, 255, 100};

    nvcv::Tensor imgIn  = test::CreateTensor(1, inWidth, inHeight, nvcv::FMT_RGBA8);
    nvcv::Tensor imgOut = test::CreateTensor(1, inWidth, inHeight, nvcv::FMT_RGBA8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*output);
    ASSERT_TRUE(outAccess);

    long inSampleStride  = inAccess->numRows() * inAccess->rowStride();
    long outSampleStride = outAccess->numRows() * outAccess->rowStride();

    int inBufSize  = inSampleStride * inAccess->numSamples();
    int outBufSize = outSampleStride * outAccess->numSamples();

    NVCVRectI bndBox = {bboxX, bboxY, bboxW, bboxH};

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0x00, inSampleStride * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0x00, outSampleStride * outAccess->numSamples()));

    std::vector<uint8_t> gold(outBufSize);
    // setGoldBuffer(gold, *outAccess, bndBox);

    // run operator
    cvcuda::BndBox op;

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, bndBox, thickness, borderColor, fillColor, MSAA));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), input->basePtr(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    EXPECT_EQ(gold, test);
}
