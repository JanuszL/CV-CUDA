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

#include "OsdUtils.cuh"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpBoxBlur.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <fstream>
#include <iostream>
#include <iterator>

namespace gt   = ::testing;
namespace test = nvcv::test;

static void setGoldBuffer(std::vector<uint8_t> &vect, const nvcv::TensorDataAccessStridedImagePlanar &data,
                          nvcv::Byte *inBuf, NVCVBlurBoxesI bboxes, cudaStream_t stream)
{
    auto context = cuosd_context_create();

    for (int n = 0; n < bboxes.batch; n++)
    {
        test::osd::Image *image = test::osd::create_image(data.numCols(), data.numRows(), test::osd::ImageFormat::RGBA);
        int               bufSize = data.numCols() * data.numRows() * data.numChannels();
        EXPECT_EQ(cudaSuccess, cudaMemcpy(image->data0, inBuf + n * bufSize, bufSize, cudaMemcpyDeviceToDevice));

        auto numBoxes = bboxes.numBoxes[n];

        for (int i = 0; i < numBoxes; i++)
        {
            auto bbox = bboxes.boxes[i];

            int left   = std::max(std::min(bbox.rect.x, data.numCols() - 1), 0);
            int top    = std::max(std::min(bbox.rect.y, data.numRows() - 1), 0);
            int right  = std::max(std::min(left + bbox.rect.width - 1, data.numCols() - 1), 0);
            int bottom = std::max(std::min(top + bbox.rect.height - 1, data.numRows() - 1), 0);

            if (left == right || top == bottom || bbox.rect.width < 3 || bbox.rect.height < 3 || bbox.kernelSize < 1)
            {
                continue;
            }

            int kernelSize = bbox.kernelSize;

            cuosd_draw_boxblur(context, left, top, right, bottom, kernelSize);
        }

        test::osd::cuosd_apply(context, image, stream);

        bboxes.boxes = (NVCVBlurBoxI *)((unsigned char *)bboxes.boxes + numBoxes * sizeof(NVCVBlurBoxI));
        EXPECT_EQ(cudaSuccess, cudaMemcpy(vect.data() + n * bufSize, image->data0, bufSize, cudaMemcpyDeviceToHost));
        test::osd::free_image(image);
    }

    cuosd_context_destroy(context);
}

// clang-format off
NVCV_TEST_SUITE_P(OpBoxBlur, test::ValueList<int, int, int, int, int, int, int, int>
{
    //  inN,    inW,    inH,    cols,   rows,   wBox,   hBox,   ks
    {   1,      224,    224,    5,      5,      16,     16,     7   },
    {   8,      224,    224,    5,      5,      16,     16,     7   },
    {   16,     224,    224,    5,      5,      16,     16,     7   },
    {   1,      1280,   720,    10,     10,     64,     64,     13  },
    {   1,      1920,   1080,   15,     15,     64,     64,     19  },
    {   1,      3840,   2160,   15,     15,     128,    128,    23  },
});

// clang-format on
TEST_P(OpBoxBlur, BoxBlur_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int inN  = GetParamValue<0>();
    int inW  = GetParamValue<1>();
    int inH  = GetParamValue<2>();
    int cols = GetParamValue<3>();
    int rows = GetParamValue<4>();
    int wBox = GetParamValue<5>();
    int hBox = GetParamValue<6>();
    int ks   = GetParamValue<7>();

    NVCVBlurBoxesI            blurBoxes;
    std::vector<int>          numBoxVec;
    std::vector<NVCVBlurBoxI> blurBoxVec;

    for (int n = 0; n < inN; n++)
    {
        numBoxVec.push_back(cols * rows);
        for (int i = 0; i < cols; i++)
        {
            int x = (inW / cols) * i + wBox / 2;
            for (int j = 0; j < rows; j++)
            {
                NVCVBlurBoxI blurBox;
                blurBox.rect.x      = x;
                blurBox.rect.y      = (inH / rows) * j + hBox / 2;
                blurBox.rect.width  = wBox;
                blurBox.rect.height = hBox;
                blurBox.kernelSize  = ks;
                blurBoxVec.push_back(blurBox);
            }
        }
    }

    blurBoxes.batch    = inN;
    blurBoxes.numBoxes = numBoxVec.data();
    blurBoxes.boxes    = blurBoxVec.data();

    nvcv::Tensor imgIn  = test::CreateTensor(inN, inW, inH, nvcv::FMT_RGBA8);
    nvcv::Tensor imgOut = test::CreateTensor(inN, inW, inH, nvcv::FMT_RGBA8);

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

    std::vector<uint8_t>          inVec(inBufSize);
    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);
    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    EXPECT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output->basePtr(), inVec.data(), outBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::BoxBlur op;

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, blurBoxes));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), input->basePtr(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    std::vector<uint8_t> gold(outBufSize);
    setGoldBuffer(gold, *inAccess, input->basePtr(), blurBoxes, stream);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    EXPECT_EQ(gold, test);
}
