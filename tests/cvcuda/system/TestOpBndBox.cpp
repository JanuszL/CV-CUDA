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
#include "OsdUtils.cuh"

namespace gt   = ::testing;
namespace test = nvcv::test;

static void setGoldBuffer(std::vector<uint8_t> &vect, const nvcv::TensorDataAccessStridedImagePlanar &data, nvcv::Byte *inBuf,
                          NVCVRectI bbox, int thickness, uchar4 borderColor, uchar4 fillColor, cudaStream_t stream)
{
    test::osd::Image* image = test::osd::create_image(data.numCols(), data.numRows(), test::osd::ImageFormat::RGBA);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(image->data0, inBuf, vect.size(), cudaMemcpyDeviceToDevice));

    test::osd::save_image(image, "workspace/input.png", stream);

    auto context = cuosd_context_create();

    int left    = bbox.x;
    int top     = bbox.y;
    int right   = left + bbox.width - 1;
    int bottom  = top + bbox.height - 1;

    cuosd_draw_rectangle(context, left, top, right, bottom, thickness,
                         {borderColor.x, borderColor.y, borderColor.z, borderColor.w},
                         {fillColor.x, fillColor.y, fillColor.z, fillColor.w});

    test::osd::cuosd_apply(context, image, stream);
    cuosd_context_destroy(context);

    EXPECT_EQ(cudaSuccess, cudaMemcpy(vect.data(), image->data0, vect.size(), cudaMemcpyDeviceToHost));
    test::osd::save_image(image, "workspace/output.png", stream);
}

// clang-format off
NVCV_TEST_SUITE_P(OpBndBox, test::ValueList<int, int, int, int, int, int, int>
{
    //      inW,    inH,    bboxX,  bboxY,  bboxW,  bboxH,  thickness
    {       600,    600,    0,      0,      200,    200,    3       },
    {       600,    600,    0,      0,      200,    200,    -1      },
    {       600,    600,    0,      0,      200,    200,    10      },
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

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0xFF, inSampleStride * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0xFF, outSampleStride * outAccess->numSamples()));

    std::vector<uint8_t> gold(outBufSize);
    setGoldBuffer(gold, *inAccess, input->basePtr(), bndBox, thickness, borderColor, fillColor, stream);

    // run operator
    cvcuda::BndBox op;

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, bndBox, thickness, borderColor, fillColor));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), input->basePtr(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    EXPECT_EQ(gold, test);
}
