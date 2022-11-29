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

//#include <common/Utils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpWarpAffine.hpp>

#include <cmath>
#include <random>

namespace nvcv     = nv::cv;
namespace nvcvcuda = nv::cv::cuda;
namespace test     = nv::cv::test;
using namespace nv::cv::cuda;

//#define DBG 1

inline int reflect_idx_low(int x, int last_col_)
{
    return (std::abs(x) - (x < 0)) % (last_col_ + 1);
}

inline int reflect_idx_high(int x, int last_col_)
{
    return (last_col_ - std::abs(last_col_ - x) + (x > last_col_));
}

inline int reflect_idx(int x, int last_col_)
{
    return reflect_idx_low(reflect_idx_high(x, last_col_), last_col_);
}

inline int reflect101_idx_low(int x, int last_col_)
{
    return std::abs(x) % (last_col_ + 1);
}

inline int reflect101_idx_high(int x, int last_col_)
{
    return std::abs(last_col_ - std::abs(last_col_ - x)) % (last_col_ + 1);
}

inline int reflect101_idx(int x, int last_col_)
{
    return reflect101_idx_low(reflect101_idx_high(x, last_col_), last_col_);
}

inline int wrap_idx_low(int x, int width_)
{
    return (x >= 0) ? x : (x - ((x - width_ + 1) / width_) * width_);
}

inline int wrap_idx_high(int x, int width_)
{
    return (x < width_) ? x : (x % width_);
}

inline int wrap_idx(int x, int width_)
{
    return wrap_idx_high(wrap_idx_low(x, width_), width_);
}

template<typename T>
T getPixel(const T *srcPtr, const int y, const int x, int k, int width, int height, int srcRowPitch,
           int elementsPerPixel, NVCVBorderType borderMode, const float4 borderVal)
{
    if (borderMode == NVCV_BORDER_CONSTANT)
    {
        return (x >= 0 && x < width && y >= 0 && y < height) ? srcPtr[y * srcRowPitch + x * elementsPerPixel + k]
                                                             : static_cast<T>(GetElement(borderVal, k));
    }
    else if (borderMode == NVCV_BORDER_REPLICATE)
    {
        int x1 = std::max(std::min(x, width - 1), 0);
        int y1 = std::max(std::min(y, height - 1), 0);
        return srcPtr[y1 * srcRowPitch + x1 * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT)
    {
        int x1 = reflect_idx(x, width - 1);
        int y1 = reflect_idx(y, height - 1);
        return srcPtr[y1 * srcRowPitch + x1 * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT101)
    {
        int x1 = reflect101_idx(x, width - 1);
        int y1 = reflect101_idx(y, height - 1);
        return srcPtr[y1 * srcRowPitch + x1 * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_WRAP)
    {
        int x1 = wrap_idx(x, width);
        int y1 = wrap_idx(y, height);
        return srcPtr[y1 * srcRowPitch + x1 * elementsPerPixel + k];
    }
    else
    {
        return 0;
    }
}

template<typename T>
static void WarpAffineGold(std::vector<uint8_t> &hDst, int dstRowPitch, nvcv::Size2D dstSize,
                           const std::vector<uint8_t> &hSrc, int srcRowPitch, nvcv::Size2D srcSize,
                           nvcv::ImageFormat fmt, const float trans_matrix[6], const int flags,
                           NVCVBorderType borderMode, const float4 borderVal)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    int srcWidth  = srcSize.w;
    int srcHeight = srcSize.h;

    const int interpolation = flags & NVCV_INTERP_MAX;

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            float src_x = (float)(dst_x * trans_matrix[0] + dst_y * trans_matrix[1] + trans_matrix[2]);
            float src_y = (float)(dst_x * trans_matrix[3] + dst_y * trans_matrix[4] + trans_matrix[5]);

            if (interpolation == NVCV_INTERP_LINEAR)
            {
                const int x1 = std::floor(src_x);
                const int y1 = std::floor(src_y);

                const int x2 = x1 + 1;
                const int y2 = y1 + 1;

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    float out = 0;

                    T src_reg = getPixel<T>(srcPtr, y1, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                            borderMode, borderVal);
                    out += src_reg * ((x2 - src_x) * (y2 - src_y));

                    src_reg = getPixel<T>(srcPtr, y1, x2, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                          borderMode, borderVal);
                    out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

                    src_reg = getPixel<T>(srcPtr, y2, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                          borderMode, borderVal);
                    out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

                    src_reg = getPixel<T>(srcPtr, y2, x2, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                          borderMode, borderVal);
                    out     = out + src_reg * ((src_x - x1) * (src_y - y1));

                    out                                                        = std::rint(out);
                    dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k] = out < 0 ? 0 : (out > 255 ? 255 : out);
                }
            }
            else if (interpolation == NVCV_INTERP_NEAREST)
            {
                const int x1 = std::trunc(src_x);
                const int y1 = std::trunc(src_y);
                for (int k = 0; k < elementsPerPixel; k++)
                {
                    T src_reg = getPixel<T>(srcPtr, y1, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                            borderMode, borderVal);
                    dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k] = src_reg;
                }
            }
            else
            {
                return;
            }
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpWarpAffine, test::ValueList<int, int, int, int, float, float, float, float, float, float, NVCVInterpolationType, NVCVBorderType, float, float, float, float, int>
{
    // srcWidth, srcHeight, dstWidth, dstHeight, transformation_matrix,       interpolation,           borderType,  borderValue, batchSize
    {        4,        4,       4,        4,      1, 0, 0, 0, 1, 0, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      1, 0, 1, 0, 1, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      2, 0, 1, 0, 2, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      0.5, 0, 1, 0, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      0.5, 1, 1, 0, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      0.5, 1, 1, 1, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         1},

    {        4,        4,       4,        4,      1, 0, 0, 0, 1, 0, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      1, 0, 1, 0, 1, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      2, 0, 1, 0, 2, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      0.5, 0, 1, 0, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         1},
    {        4,        4,       4,        4,      0.5, 1, 1, 0, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         1},

    {        4,        4,       4,        4,      1, 0, 0, 0, 1, 0, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         4},
    {        4,        5,       4,        5,      1, 0, 0, 0, 1, 0, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         4},
    {        5,        4,       5,        4,      1, 0, 0, 0, 1, 0, NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,   0, 0, 0, 0,         4},

    {        5,        4,       5,        4,      0.5, 1, 1, 0, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_REFLECT,   0, 0, 0, 0,         4},
    {        5,        4,       5,        4,      0.5, 1, 1, 0, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_REFLECT101,   0, 0, 0, 0,         4},
    {        5,        4,       5,        4,      0.5, 1, 1, 0, 0.5, 1, NVCV_INTERP_LINEAR, NVCV_BORDER_WRAP,   0, 0, 0, 0,         4},

    {        5,        4,       5,        4,      1, 0, 0, 0, 1, 0, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},
    {        5,        4,       5,        4,      1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},

    {        5,        4,       5,        4,     1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},
    {        5,        4,       5,        4,     1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST, NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4},
    {        5,        4,       5,        4,     1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST, NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4},
    {        5,        4,       5,        4,     1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST, NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4},
    {        5,        4,       5,        4,     1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST, NVCV_BORDER_WRAP,   1, 2, 3, 4,         4},

    {        5,        4,       5,        4,     0.5, 0, 0, 0, 0.5, 0, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},
    {        5,        4,       5,        4,      0.5, 1, 1, 0, 0.5, 1, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},
    {        5,        4,       5,        4,      0.5, 1, 1, 0, 0.5, 1, NVCV_INTERP_NEAREST, NVCV_BORDER_WRAP,   0, 0, 0, 0,         4},

    // vary output size
    {        4,        4,       8,        8,      0.5, 1, 1, 1, 0.5, 1, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},
    {        4,        4,       8,        8,      0.5, 1, 0, 1, 0.5, 0, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},
    {        4,        4,       8,        8,      0.5, 0, 0, 0, 0.5, 0, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4},
    {        2,        2,       4,        4,      0, 1, 0, 1, 0, 0, NVCV_INTERP_NEAREST, NVCV_BORDER_CONSTANT,   0, 0, 0, 0,         4},

});

// clang-format on

TEST_P(OpWarpAffine, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    const float trans_matrix[6] = {GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
                                   GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>()};

    NVCVInterpolationType interpolation = GetParamValue<10>();

    NVCVBorderType borderMode = GetParamValue<11>();

    const float4 borderValue = {GetParamValue<12>(), GetParamValue<13>(), GetParamValue<14>(), GetParamValue<15>()};

    int numberOfImages = GetParamValue<16>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    const nvcv::Size2D dsize = {dstWidth, dstHeight};

    const int flags = interpolation;

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowPitch = srcWidth * fmt.planePixelStrideBytes(0);

    std::default_random_engine randEng;

    for (int i = 0; i < numberOfImages; ++i)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcVecRowPitch);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowPitchBytes(), srcVec[i].data(), srcVecRowPitch,
                               srcVecRowPitch, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    nv::cvop::WarpAffine warpAffineOp;
    EXPECT_NO_THROW(warpAffineOp(stream, imgSrc, imgDst, trans_matrix, dsize, flags, borderMode, borderValue));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowPitch = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecRowPitch);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowPitch, dstAccess->sampleData(i), dstAccess->rowPitchBytes(),
                               dstVecRowPitch, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowPitch);

        // Generate gold result
        WarpAffineGold<uint8_t>(goldVec, dstVecRowPitch, {dstWidth, dstHeight}, srcVec[i], srcVecRowPitch,
                                {srcWidth, srcHeight}, fmt, trans_matrix, flags, borderMode, borderValue);

#if DBG
        std::cout << "\nPrint src vec " << std::endl;
        for (int k = 0; k < srcHeight; k++)
        {
            for (int j = 0; j < srcVecRowPitch; j++)
            {
                std::cout << static_cast<int>(srcVec[i][k * srcVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowPitch; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint warped output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowPitch; j++)
            {
                std::cout << static_cast<int>(testVec[k * dstVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }
#endif

        EXPECT_EQ(goldVec, testVec);
        //EXPECT_EQ(1,1);
    }
}
