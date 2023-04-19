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
#include <cvcuda/OpBndBox.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>

#include <fstream>
#include <iterator>

namespace gt   = ::testing;
namespace test = nvcv::test;

static int randl(int l, int h)
{
    int value = rand() % (h - l + 1);
    return l + value;
}

static void loadGoldBuffer(std::vector<uint8_t> &vect, std::string goldPath)
{
    std::ifstream input(goldPath.c_str(), std::ios::binary);
    vect = std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}

static void dumpGoldBuffer(std::vector<uint8_t> &vect, const nvcv::TensorDataAccessStridedImagePlanar &data,
                          nvcv::Byte *inBuf, NVCVBndBoxesI bboxes, cudaStream_t stream, std::string goldPath)
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

            if (left == right || top == bottom || bbox.rect.width <= 0 || bbox.rect.height <= 0)
            {
                continue;
            }

            int thickness = bbox.thickness;

            cuOSDColor borderColor = {bbox.borderColor.r, bbox.borderColor.g, bbox.borderColor.b, bbox.borderColor.a};
            cuOSDColor fillColor   = {bbox.fillColor.r, bbox.fillColor.g, bbox.fillColor.b, bbox.fillColor.a};

            cuosd_draw_rectangle(context, left, top, right, bottom, thickness, borderColor, fillColor);
        }

        test::osd::cuosd_apply(context, image, stream);

        bboxes.boxes = (NVCVBndBoxI *)((unsigned char *)bboxes.boxes + numBoxes * sizeof(NVCVBndBoxI));
        EXPECT_EQ(cudaSuccess, cudaMemcpy(vect.data() + n * bufSize, image->data0, bufSize, cudaMemcpyDeviceToHost));
        test::osd::free_image(image);
    }

    cuosd_context_destroy(context);

    std::ofstream output(goldPath.c_str(), std::ios::out | std::ios::binary);
    output.write((const char *)vect.data(), vect.size());
    output.close();
}

static void dumpTest(std::vector<uint8_t> &vect, const nvcv::TensorDataAccessStridedImagePlanar &data,
                     nvcv::Byte *testBuf)
{
    for (int n = 0; n < data.numSamples(); n++)
    {
        test::osd::Image *image = test::osd::create_image(data.numCols(), data.numRows(), test::osd::ImageFormat::RGBA);
        int               bufSize = data.numCols() * data.numRows() * data.numChannels();
        EXPECT_EQ(cudaSuccess, cudaMemcpy(image->data0, testBuf + n * bufSize, bufSize, cudaMemcpyDeviceToDevice));
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpBndBox, test::ValueList<int, int, int, int, int, std::string>
{
    //  inN,    inW,    inH,    num,    seed,   goldPath
    {   1,      224,    224,    100,    3,      "workspace/GoldBndBox0.bin"   },
    {   8,      224,    224,    100,    7,      "workspace/GoldBndBox1.bin"   },
    {   16,     224,    224,    100,    11,     "workspace/GoldBndBox2.bin"   },
    {   1,      1280,   720,    100,    23,     "workspace/GoldBndBox3.bin"   },
    {   1,      1920,   1080,   200,    37,     "workspace/GoldBndBox4.bin"   },
    {   1,      3840,   2160,   200,    59,     "workspace/GoldBndBox5.bin"   },
});

// clang-format on
TEST_P(OpBndBox, BndBox_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int inN              = GetParamValue<0>();
    int inW              = GetParamValue<1>();
    int inH              = GetParamValue<2>();
    int num              = GetParamValue<3>();
    int sed              = GetParamValue<4>();
    std::string goldPath = GetParamValue<5>();

    NVCVBndBoxesI            bndBoxes;
    std::vector<int>         numBoxVec;
    std::vector<NVCVBndBoxI> bndBoxVec;

    srand(sed);
    for (int n = 0; n < inN; n++)
    {
        numBoxVec.push_back(num);
        for (int i = 0; i < num; i++)
        {
            NVCVBndBoxI bndBox;
            bndBox.rect.x      = randl(0, inW - 1);
            bndBox.rect.y      = randl(0, inH - 1);
            bndBox.rect.width  = randl(1, inW);
            bndBox.rect.height = randl(1, inH);
            bndBox.thickness   = randl(-1, 30);
            bndBox.fillColor   = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
            bndBox.borderColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
            bndBoxVec.push_back(bndBox);
        }
    }

    bndBoxes.batch    = inN;
    bndBoxes.numBoxes = numBoxVec.data();
    bndBoxes.boxes    = bndBoxVec.data();

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

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0xFF, inSampleStride * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0xFF, outSampleStride * outAccess->numSamples()));

    // run operator
    cvcuda::BndBox op;

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, bndBoxes));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), input->basePtr(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    std::vector<uint8_t> gold(outBufSize);
    // dumpGoldBuffer(gold, *inAccess, input->basePtr(), bndBoxes, stream, goldPath);
    loadGoldBuffer(gold, goldPath);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    dumpTest(gold, *outAccess, output->basePtr());
    EXPECT_EQ(gold, test);
}
