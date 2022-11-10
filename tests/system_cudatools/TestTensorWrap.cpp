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

#include "DeviceTensorWrap.hpp" // to test in device

#include <common/TypedTests.hpp> // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/Tensor.hpp>       // for Tensor, etc.
#include <nvcv/cuda/MathOps.hpp> // for operator == to allow EXPECT_EQ

#include <limits>

namespace cuda  = nv::cv::cuda;
namespace ttype = nv::cv::test::type;

// -------------------------- Testing Tensor2DWrap -----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<PackedImage<int, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedImage<short3, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<PackedImage<float1, 2, 4>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedImage<uchar4, 3, 3>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using PixelType = typename InputType::value_type;

    cuda::Tensor2DWrap<const PixelType> wrap(input.data(), input.pitchBytes);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], input.pitchBytes);

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int y = 0; y < input.height; ++y)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(y))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(y))>>);

        EXPECT_EQ(wrap.ptr(y), &input[y * input.width]);

        for (int x = 0; x < input.width; ++x)
        {
            EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(y, x))>);
            EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(y, x))>>);

            EXPECT_EQ(wrap.ptr(y, x), &input[y * input.width + x]);

            int2 c2{x, y};

            EXPECT_TRUE(std::is_reference_v<decltype(wrap[c2])>);
            EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c2])>>);

            EXPECT_EQ(wrap[c2], input[y * input.width + x]);
        }
    }
}

TYPED_TEST(Tensor2DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensor2DWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<PackedImage<int, 2, 2>{}>, ttype::Value<PackedImage<int, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedImage<short3, 1, 2>{}>, ttype::Value<PackedImage<short3, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<PackedImage<float1, 2, 4>{}>, ttype::Value<PackedImage<float1, 2, 4>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedImage<uchar4, 3, 3>{}>, ttype::Value<PackedImage<uchar4, 3, 3>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using PixelType = typename InputType::value_type;

    cuda::Tensor2DWrap<PixelType> wrap(test.data(), test.pitchBytes);

    for (int y = 0; y < gold.height; ++y)
    {
        for (int x = 0; x < gold.width; ++x)
        {
            int2 c{x, y};

            wrap[c] = gold[y * gold.width + x];
        }
    }

    EXPECT_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapImageWrapTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_S32>, ttype::Value<PackedImage<int, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_2S16>, ttype::Value<PackedImage<short2, 1, 2>{
        short2{-12, 2}, short2{5678, -2345}}>>,
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_F32>, ttype::Value<PackedImage<float1, 2, 4>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>, ttype::Value<PackedImage<uchar4, 3, 3>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapImageWrapTest, correct_with_image_wrap)
{
    auto imgFormat = ttype::GetValue<TypeParam, 0>;
    auto input     = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using PixelType = typename InputType::value_type;

    nv::cv::ImageDataPitchDevice::Buffer buf;
    buf.numPlanes            = 1;
    buf.planes[0].width      = input.width;
    buf.planes[0].height     = input.height;
    buf.planes[0].pitchBytes = input.pitchBytes;
    buf.planes[0].buffer     = reinterpret_cast<void *>(input.data());

    nv::cv::ImageWrapData img{
        nv::cv::ImageDataPitchDevice{nv::cv::ImageFormat{imgFormat}, buf}
    };

    auto *dev = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(img.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor2DWrap<PixelType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], input.pitchBytes);

    for (int y = 0; y < input.height; ++y)
    {
        for (int x = 0; x < input.width; ++x)
        {
            int2 c2{x, y};

            EXPECT_EQ(wrap[c2],
                      *reinterpret_cast<PixelType *>((reinterpret_cast<uint8_t *>(dev->plane(0).buffer)
                                                      + y * dev->plane(0).pitchBytes + x * sizeof(PixelType))));
        }
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor2DWrapImageTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_IMAGE_FORMAT_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_IMAGE_FORMAT_Y8>>,
    ttype::Types<const short2, ttype::Value<NVCV_IMAGE_FORMAT_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_IMAGE_FORMAT_RGB8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>>,
    ttype::Types<float3, ttype::Value<NVCV_IMAGE_FORMAT_RGBf32>>,
    ttype::Types<const float4, ttype::Value<NVCV_IMAGE_FORMAT_RGBAf32>>
>);

// clang-format on

TYPED_TEST(Tensor2DWrapImageTest, correct_with_image)
{
    using PixelType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Image img({213, 211}, nv::cv::ImageFormat{imgFormat});

    const auto *dev = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(img.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor2DWrap<PixelType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], dev->plane(0).pitchBytes);

    const PixelType *ptr0 = reinterpret_cast<const PixelType *>(dev->plane(0).buffer);
    const PixelType *ptr1 = reinterpret_cast<const PixelType *>(reinterpret_cast<const uint8_t *>(dev->plane(0).buffer)
                                                                + dev->plane(0).pitchBytes);

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
}

TYPED_TEST(Tensor2DWrapImageTest, it_works_in_device)
{
    using PixelType   = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat    = ttype::GetValue<TypeParam, 1>;
    using ChannelType = cuda::BaseType<PixelType>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nv::cv::Image img({357, 642}, nv::cv::ImageFormat{imgFormat});

    int width    = img.size().w;
    int height   = img.size().h;
    int channels = img.format().numChannels();

    const auto *dev = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(img.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor2DWrap<PixelType> wrap(*dev);

    DeviceSetOnes(wrap, {width, height}, stream);

    int    pitchBytes    = dev->plane(0).pitchBytes;
    int    pitchElements = pitchBytes / sizeof(ChannelType);
    size_t sizeBytes     = height * pitchBytes;
    size_t sizeElements  = height * pitchElements;

    std::vector<ChannelType> test(sizeElements);
    std::vector<ChannelType> gold(sizeElements);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < channels; k++)
            {
                gold[i * pitchElements + j * channels + k] = ChannelType{1};
            }
        }
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->plane(0).buffer, sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// --------------------------- Testing Tensor3DWrap ----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedTensor<short3, 2, 2, 1>{
        short3{-12, 2, -34}, short3{5678, -2345, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}}>>,
    ttype::Types<ttype::Value<PackedTensor<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using PixelType = typename InputType::value_type;

    cuda::Tensor3DWrap<const PixelType> wrap(input.data(), input.pitchBytes1, input.pitchBytes2);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], input.pitchBytes1);
    EXPECT_EQ(pitchBytes[1], input.pitchBytes2);

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int b = 0; b < input.batches; ++b)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b))>>);

        EXPECT_EQ(wrap.ptr(b), &input[b * input.width * input.height]);

        for (int y = 0; y < input.height; ++y)
        {
            EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y))>);
            EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y))>>);

            EXPECT_EQ(wrap.ptr(b, y), &input[b * input.width * input.height + y * input.width]);

            for (int x = 0; x < input.width; ++x)
            {
                EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y, x))>);
                EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y, x))>>);

                EXPECT_EQ(wrap.ptr(b, y, x), &input[b * input.width * input.height + y * input.width + x]);

                int3 c3{x, y, b};

                EXPECT_TRUE(std::is_reference_v<decltype(wrap[c3])>);
                EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c3])>>);

                EXPECT_EQ(wrap[c3], input[b * input.width * input.height + y * input.width + x]);
            }
        }
    }
}

TYPED_TEST(Tensor3DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensor3DWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor<int, 1, 2, 2>{}>, ttype::Value<PackedTensor<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedTensor<short3, 2, 2, 1>{}>, ttype::Value<PackedTensor<short3, 2, 2, 1>{
        short3{-12, 2, -34}, short3{5678, -2345, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}}>>,
    ttype::Types<ttype::Value<PackedTensor<float1, 2, 2, 2>{}>, ttype::Value<PackedTensor<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor<uchar4, 3, 3, 1>{}>, ttype::Value<PackedTensor<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using PixelType = typename InputType::value_type;

    cuda::Tensor3DWrap<PixelType> wrap(test.data(), test.pitchBytes1, test.pitchBytes2);

    for (int b = 0; b < test.batches; ++b)
    {
        for (int y = 0; y < test.height; ++y)
        {
            for (int x = 0; x < test.width; ++x)
            {
                int3 c{x, y, b};

                wrap[c] = gold[b * test.height * test.width + y * test.width + x];
            }
        }
    }

    EXPECT_EQ(test, gold);
}

// The death tests below are to be run in debug mode only

#ifndef NDEBUG

TEST(Tensor3DWrapBigPitchDeathTest, it_dies)
{
    using DataType = uint8_t;
    int64_t height = 2;
    int64_t width  = std::numeric_limits<int>::max();

    nv::cv::PixelType                     dt{NVCV_PIXEL_TYPE_U8};
    nv::cv::TensorDataPitchDevice::Buffer buf;
    buf.pitchBytes[2] = sizeof(DataType);
    buf.pitchBytes[1] = width * buf.pitchBytes[2];
    buf.pitchBytes[0] = height * buf.pitchBytes[1];
    buf.data          = reinterpret_cast<void *>(123);

    nv::cv::TensorWrapData tensor{
        nv::cv::TensorDataPitchDevice{nv::cv::TensorShape{{1, height, width}, "NHW"}, dt, buf}
    };

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    EXPECT_DEATH({ cuda::Tensor3DWrap<DataType> wrap(*dev); }, "");
}

#endif

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapTensorWrapTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_S32>, ttype::Value<PackedTensor<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_2S16>, ttype::Value<PackedTensor<short2, 2, 2, 1>{
        short2{-12, 2}, short2{5678, -2345},
        short2{123, 0}, short2{-9876, 4321}}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_F32>, ttype::Value<PackedTensor<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_4U8>, ttype::Value<PackedTensor<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapTensorWrapTest, correct_with_tensor_wrap)
{
    auto dataType = ttype::GetValue<TypeParam, 0>;
    auto input    = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using PixelType = typename InputType::value_type;

    int n = input.batches;
    int h = input.height;
    int w = input.width;

    nv::cv::TensorDataPitchDevice::Buffer buf;
    buf.pitchBytes[0] = input.pitchBytes1;
    buf.pitchBytes[1] = input.pitchBytes2;
    buf.pitchBytes[2] = input.pitchBytes3;
    buf.data          = reinterpret_cast<void *>(input.data());

    nv::cv::TensorWrapData tensor{
        nv::cv::TensorDataPitchDevice{nv::cv::TensorShape{{n, h, w}, "NHW"}, nv::cv::PixelType{dataType}, buf}
    };

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<PixelType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], input.pitchBytes1);
    EXPECT_EQ(pitchBytes[1], input.pitchBytes2);

    for (int b = 0; b < input.batches; ++b)
    {
        for (int y = 0; y < input.height; ++y)
        {
            for (int x = 0; x < input.width; ++x)
            {
                int3 c3{x, y, b};

                EXPECT_EQ(wrap[c3], *reinterpret_cast<PixelType *>((reinterpret_cast<uint8_t *>(dev->data())
                                                                    + b * dev->pitchBytes(0) + y * dev->pitchBytes(1)
                                                                    + x * dev->pitchBytes(2))));
            }
        }
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor3DWrapTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_IMAGE_FORMAT_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_IMAGE_FORMAT_Y8>>,
    ttype::Types<const short2, ttype::Value<NVCV_IMAGE_FORMAT_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_IMAGE_FORMAT_RGB8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_IMAGE_FORMAT_RGBA8>>,
    ttype::Types<float3, ttype::Value<NVCV_IMAGE_FORMAT_RGBf32>>,
    ttype::Types<const float4, ttype::Value<NVCV_IMAGE_FORMAT_RGBAf32>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapTensorTest, correct_with_tensor)
{
    using PixelType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Tensor tensor(3, {213, 211}, nv::cv::ImageFormat{imgFormat});

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<PixelType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], dev->pitchBytes(0));
    EXPECT_EQ(pitchBytes[1], dev->pitchBytes(1));

    const PixelType *ptr0 = reinterpret_cast<const PixelType *>(dev->data());
    const PixelType *ptr1
        = reinterpret_cast<const PixelType *>(reinterpret_cast<const uint8_t *>(dev->data()) + dev->pitchBytes(0));
    const PixelType *ptr12 = reinterpret_cast<const PixelType *>(reinterpret_cast<const uint8_t *>(dev->data())
                                                                 + dev->pitchBytes(0) + 2 * dev->pitchBytes(1));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
    EXPECT_EQ(wrap.ptr(1, 2), ptr12);
}

TYPED_TEST(Tensor3DWrapTensorTest, it_works_in_device)
{
    using PixelType   = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat    = ttype::GetValue<TypeParam, 1>;
    using ChannelType = cuda::BaseType<PixelType>;

    nv::cv::Tensor tensor(4, {357, 642}, nv::cv::ImageFormat{imgFormat});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int batches  = tensor.shape()[0];
    int height   = tensor.shape()[1];
    int width    = tensor.shape()[2];
    int channels = tensor.shape()[3];

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<PixelType> wrap(*dev);

    DeviceSetOnes(wrap, {width, height, batches}, stream);

    int    imgPitchElements = dev->pitchBytes(0) / sizeof(ChannelType);
    int    rowPitchElements = dev->pitchBytes(1) / sizeof(ChannelType);
    int    pixPitchElements = dev->pitchBytes(2) / sizeof(ChannelType);
    size_t sizeBytes        = batches * dev->pitchBytes(0);
    size_t sizeElements     = batches * dev->pitchBytes(0) / sizeof(ChannelType);

    std::vector<ChannelType> test(sizeElements);
    std::vector<ChannelType> gold(sizeElements);

    for (int b = 0; b < batches; b++)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < channels; k++)
                {
                    gold[b * imgPitchElements + i * rowPitchElements + j * pixPitchElements + k] = ChannelType{1};
                }
            }
        }
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->data(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}
