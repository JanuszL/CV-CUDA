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

// -------------------------- Testing Tensor1DWrap -----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor1DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<std::array<int, 2>{-5, 1}>>,
    ttype::Types<ttype::Value<std::array<short3, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<std::array<float1, 4>{
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<std::array<uchar4, 3>{
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor1DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::Tensor1DWrap<const ValueType> wrap(input.data());

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int i = 0; i < static_cast<int>(input.size()); ++i)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(i))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(i))>>);

        EXPECT_EQ(wrap.ptr(i), &input[i]);

        int1 c1{i};

        EXPECT_TRUE(std::is_reference_v<decltype(wrap[c1])>);
        EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c1])>>);

        EXPECT_EQ(wrap[c1], input[i]);
    }
}

TYPED_TEST(Tensor1DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensor1DWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor1DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<std::array<int, 2>{}>, ttype::Value<std::array<int, 2>{
       -5, 1}>>,
    ttype::Types<ttype::Value<std::array<short3, 2>{}>, ttype::Value<std::array<short3, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}}>>,
    ttype::Types<ttype::Value<std::array<float1, 4>{}>, ttype::Value<std::array<float1, 4>{
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<std::array<uchar4, 3>{}>, ttype::Value<std::array<uchar4, 3>{
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor1DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::Tensor1DWrap<ValueType> wrap(test.data());

    for (int i = 0; i < static_cast<int>(gold.size()); ++i)
    {
        int1 c{i};

        wrap[c] = gold[i];
    }

    EXPECT_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor1DWrapTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_PIXEL_TYPE_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_PIXEL_TYPE_U8>>,
    ttype::Types<const short2, ttype::Value<NVCV_PIXEL_TYPE_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_PIXEL_TYPE_3U8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_PIXEL_TYPE_4U8>>,
    ttype::Types<float3, ttype::Value<NVCV_PIXEL_TYPE_3F32>>,
    ttype::Types<const float4, ttype::Value<NVCV_PIXEL_TYPE_4F32>>
>);

// clang-format on

TYPED_TEST(Tensor1DWrapTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto pixelType  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Tensor tensor({{123}, "N"}, nv::cv::PixelType{pixelType});

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor1DWrap<ValueType> wrap(*dev);

    const ValueType *ptr0 = reinterpret_cast<const ValueType *>(dev->data());
    const ValueType *ptr1
        = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->data()) + dev->pitchBytes(0));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
}

TYPED_TEST(Tensor1DWrapTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto pixelType  = ttype::GetValue<TypeParam, 1>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nv::cv::Tensor tensor({{123}, "N"}, nv::cv::PixelType{pixelType});

    int1 size{static_cast<int>(tensor.shape()[0])};

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor1DWrap<ValueType> wrap(*dev);

    DeviceSetOnes(wrap, size, stream);

    long        pitchBytes{dev->pitchBytes(0)};
    std::size_t sizeBytes = size.x * pitchBytes;

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int i = 0; i < size.x; i++)
    {
        *reinterpret_cast<ValueType *>(&gold[i * pitchBytes]) = cuda::SetAll<ValueType>(1);
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->data(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

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
    using ValueType = typename InputType::value_type;

    cuda::Tensor2DWrap<const ValueType> wrap(input.data(), input.pitchBytes);

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
    using ValueType = typename InputType::value_type;

    cuda::Tensor2DWrap<ValueType> wrap(test.data(), test.pitchBytes);

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
    using ValueType = typename InputType::value_type;

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

    cuda::Tensor2DWrap<ValueType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], input.pitchBytes);

    for (int y = 0; y < input.height; ++y)
    {
        for (int x = 0; x < input.width; ++x)
        {
            int2 c2{x, y};

            EXPECT_EQ(wrap[c2],
                      *reinterpret_cast<ValueType *>((reinterpret_cast<uint8_t *>(dev->plane(0).buffer)
                                                      + y * dev->plane(0).pitchBytes + x * sizeof(ValueType))));
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
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Image img({213, 211}, nv::cv::ImageFormat{imgFormat});

    const auto *dev = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(img.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor2DWrap<ValueType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], dev->plane(0).pitchBytes);

    const ValueType *ptr0 = reinterpret_cast<const ValueType *>(dev->plane(0).buffer);
    const ValueType *ptr1 = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->plane(0).buffer)
                                                                + dev->plane(0).pitchBytes);

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
}

TYPED_TEST(Tensor2DWrapImageTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nv::cv::Image img({357, 642}, nv::cv::ImageFormat{imgFormat});

    int width  = img.size().w;
    int height = img.size().h;

    const auto *dev = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(img.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor2DWrap<ValueType> wrap(*dev);

    DeviceSetOnes(wrap, {width, height}, stream);

    long2       pitches{dev->plane(0).pitchBytes, sizeof(ValueType)};
    std::size_t sizeBytes = height * pitches.x;

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            *reinterpret_cast<ValueType *>(&gold[i * pitches.x + j * pitches.y]) = cuda::SetAll<ValueType>(1);
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
    ttype::Types<ttype::Value<PackedTensor3D<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedTensor3D<short3, 2, 2, 1>{
        short3{-12, 2, -34}, short3{5678, -2345, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor3DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::Tensor3DWrap<const ValueType> wrap(input.data(), input.pitchBytes1, input.pitchBytes2);

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
    ttype::Types<ttype::Value<PackedTensor3D<int, 1, 2, 2>{}>, ttype::Value<PackedTensor3D<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<PackedTensor3D<short3, 2, 2, 1>{}>, ttype::Value<PackedTensor3D<short3, 2, 2, 1>{
        short3{-12, 2, -34}, short3{5678, -2345, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<float1, 2, 2, 2>{}>, ttype::Value<PackedTensor3D<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{}>, ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{
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
    using ValueType = typename InputType::value_type;

    cuda::Tensor3DWrap<ValueType> wrap(test.data(), test.pitchBytes1, test.pitchBytes2);

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
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_S32>, ttype::Value<PackedTensor3D<int, 1, 2, 2>{
        2, 3,
       -5, 1}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_2S16>, ttype::Value<PackedTensor3D<short2, 2, 2, 1>{
        short2{-12, 2}, short2{5678, -2345},
        short2{123, 0}, short2{-9876, 4321}}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_F32>, ttype::Value<PackedTensor3D<float1, 2, 2, 2>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_4U8>, ttype::Value<PackedTensor3D<uchar4, 3, 3, 1>{
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
    using ValueType = typename InputType::value_type;

    int n = input.batches;
    int h = input.height;
    int w = input.width;

    nv::cv::TensorDataPitchDevice::Buffer buf;
    buf.pitchBytes[0] = input.pitchBytes1;
    buf.pitchBytes[1] = input.pitchBytes2;
    buf.pitchBytes[2] = sizeof(ValueType);
    buf.data          = reinterpret_cast<void *>(input.data());

    nv::cv::TensorWrapData tensor{
        nv::cv::TensorDataPitchDevice{nv::cv::TensorShape{{n, h, w}, "NHW"}, nv::cv::PixelType{dataType}, buf}
    };

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<ValueType> wrap(*dev);

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

                EXPECT_EQ(wrap[c3], *reinterpret_cast<ValueType *>((reinterpret_cast<uint8_t *>(dev->data())
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
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Tensor tensor(3, {213, 211}, nv::cv::ImageFormat{imgFormat});

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<ValueType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], dev->pitchBytes(0));
    EXPECT_EQ(pitchBytes[1], dev->pitchBytes(1));

    const ValueType *ptr0 = reinterpret_cast<const ValueType *>(dev->data());
    const ValueType *ptr1
        = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->data()) + dev->pitchBytes(0));
    const ValueType *ptr12 = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->data())
                                                                 + dev->pitchBytes(0) + 2 * dev->pitchBytes(1));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
    EXPECT_EQ(wrap.ptr(1, 2), ptr12);
}

TYPED_TEST(Tensor3DWrapTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto imgFormat  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Tensor tensor(5, {234, 567}, nv::cv::ImageFormat{imgFormat});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int batches = tensor.shape()[0];
    int height  = tensor.shape()[1];
    int width   = tensor.shape()[2];

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor3DWrap<ValueType> wrap(*dev);

    DeviceSetOnes(wrap, {width, height, batches}, stream);

    long3       pitches{dev->pitchBytes(0), dev->pitchBytes(1), dev->pitchBytes(2)};
    std::size_t sizeBytes = batches * pitches.x;

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int b = 0; b < batches; b++)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                *reinterpret_cast<ValueType *>(&gold[b * pitches.x + i * pitches.y + j * pitches.z])
                    = cuda::SetAll<ValueType>(1);
            }
        }
    }

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dev->data(), sizeBytes, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test, gold);
}

// --------------------------- Testing Tensor4DWrap ----------------------------

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{
        2, 3, 4, 5
       -5, 1, 6, 7}>>,
    ttype::Types<ttype::Value<PackedTensor4D<short3, 2, 2, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}, short3{-1, -2, -3}, short3{-567, 234, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}, short3{1, 2, 3}, short3{-56, 23, 1}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapTest, correct_content_and_is_const)
{
    auto input = ttype::GetValue<TypeParam, 0>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    cuda::Tensor4DWrap<const ValueType> wrap(input.data(), input.pitchBytes1, input.pitchBytes2, input.pitchBytes3);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], input.pitchBytes1);
    EXPECT_EQ(pitchBytes[1], input.pitchBytes2);
    EXPECT_EQ(pitchBytes[2], input.pitchBytes3);

    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr())>);
    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr())>>);

    EXPECT_EQ(wrap.ptr(), input.data());

    for (int b = 0; b < input.batches; ++b)
    {
        EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b))>);
        EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b))>>);

        EXPECT_EQ(wrap.ptr(b), &input[b * input.height * input.width * input.channels]);

        for (int y = 0; y < input.height; ++y)
        {
            EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y))>);
            EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y))>>);

            EXPECT_EQ(wrap.ptr(b, y),
                      &input[b * input.height * input.width * input.channels + y * input.width * input.channels]);

            for (int x = 0; x < input.width; ++x)
            {
                EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y, x))>);
                EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y, x))>>);

                EXPECT_EQ(wrap.ptr(b, y, x), &input[b * input.height * input.width * input.channels
                                                    + y * input.width * input.channels + x * input.channels]);

                for (int k = 0; k < input.channels; ++k)
                {
                    EXPECT_TRUE(std::is_pointer_v<decltype(wrap.ptr(b, y, x, k))>);
                    EXPECT_TRUE(std::is_const_v<std::remove_pointer_t<decltype(wrap.ptr(b, y, x, k))>>);

                    EXPECT_EQ(wrap.ptr(b, y, x, k),
                              &input[b * input.height * input.width * input.channels + y * input.width * input.channels
                                     + x * input.channels + k]);

                    int4 c4{k, x, y, b};

                    EXPECT_TRUE(std::is_reference_v<decltype(wrap[c4])>);
                    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(wrap[c4])>>);

                    EXPECT_EQ(wrap[c4], input[b * input.height * input.width * input.channels
                                              + y * input.width * input.channels + x * input.channels + k]);
                }
            }
        }
    }
}

TYPED_TEST(Tensor4DWrapTest, it_works_in_device)
{
    auto gold = ttype::GetValue<TypeParam, 0>;

    DeviceUseTensor4DWrap(gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapCopyTest, ttype::Types<
    ttype::Types<ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{}>, ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{
        2, 3, 4, 5,
       -5, 1, 6, 7}>>,
    ttype::Types<ttype::Value<PackedTensor4D<short3, 2, 2, 1, 2>{}>, ttype::Value<PackedTensor4D<short3, 2, 2, 1, 2>{
        short3{-12, 2, -34}, short3{5678, -2345, 0}, short3{-1, -2, -3}, short3{-567, 234, 0},
        short3{121, -2, 33}, short3{-876, 4321, 21}, short3{1, 2, 3}, short3{-56, 23, 1}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{}>, ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{}>, ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapCopyTest, can_change_content)
{
    auto test = ttype::GetValue<TypeParam, 0>;
    auto gold = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(test);
    using ValueType = typename InputType::value_type;

    cuda::Tensor4DWrap<ValueType> wrap(test.data(), test.pitchBytes1, test.pitchBytes2, test.pitchBytes3);

    for (int b = 0; b < test.batches; ++b)
    {
        for (int y = 0; y < test.height; ++y)
        {
            for (int x = 0; x < test.width; ++x)
            {
                for (int k = 0; k < test.channels; ++k)
                {
                    int4 c{k, x, y, b};

                    wrap[c] = gold[b * test.height * test.width * test.channels + y * test.width * test.channels
                                   + x * test.channels + k];
                }
            }
        }
    }

    EXPECT_EQ(test, gold);
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapTensorWrapTest, ttype::Types<
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_S32>, ttype::Value<PackedTensor4D<int, 1, 2, 2, 2>{
        2, 3, 4, 5,
       -5, 1, 6, 7}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_2S16>, ttype::Value<PackedTensor4D<short2, 2, 2, 1, 2>{
        short2{-12, 2}, short2{5678, -2345}, short2{123, -321}, short2{-567, 234},
        short2{123, 0}, short2{-9876, 4321}, short2{12, -32}, short2{567, -234}}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_F32>, ttype::Value<PackedTensor4D<float1, 2, 2, 2, 1>{
        float1{-1.23f}, float1{2.3f}, float1{3.45f}, float1{-4.5f},
        float1{1.23f}, float1{-2.3f}, float1{-3.45f}, float1{4.5f}}>>,
    ttype::Types<ttype::Value<NVCV_PIXEL_TYPE_4U8>, ttype::Value<PackedTensor4D<uchar4, 3, 3, 1, 1>{
        uchar4{0, 127, 231, 32}, uchar4{56, 255, 1, 2}, uchar4{42, 3, 5, 7},
        uchar4{12, 17, 230, 31}, uchar4{57, 254, 8, 1}, uchar4{41, 2, 4, 6},
        uchar4{0, 128, 233, 33}, uchar4{55, 253, 9, 1}, uchar4{40, 1, 3, 5}}>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapTensorWrapTest, correct_with_tensor_wrap)
{
    auto dataType = ttype::GetValue<TypeParam, 0>;
    auto input    = ttype::GetValue<TypeParam, 1>;

    using InputType = decltype(input);
    using ValueType = typename InputType::value_type;

    int n = input.batches;
    int h = input.height;
    int w = input.width;
    int c = input.channels;

    nv::cv::TensorDataPitchDevice::Buffer buf;
    buf.pitchBytes[0] = input.pitchBytes1;
    buf.pitchBytes[1] = input.pitchBytes2;
    buf.pitchBytes[2] = input.pitchBytes3;
    buf.pitchBytes[3] = sizeof(ValueType);
    buf.data          = reinterpret_cast<void *>(input.data());

    nv::cv::TensorWrapData tensor{
        nv::cv::TensorDataPitchDevice{nv::cv::TensorShape{{n, h, w, c}, "NHWC"}, nv::cv::PixelType{dataType}, buf}
    };

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor4DWrap<ValueType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], input.pitchBytes1);
    EXPECT_EQ(pitchBytes[1], input.pitchBytes2);
    EXPECT_EQ(pitchBytes[2], input.pitchBytes3);

    for (int b = 0; b < input.batches; ++b)
    {
        for (int y = 0; y < input.height; ++y)
        {
            for (int x = 0; x < input.width; ++x)
            {
                for (int k = 0; k < input.channels; ++k)
                {
                    int4 c4{k, x, y, b};

                    EXPECT_EQ(wrap[c4],
                              *reinterpret_cast<ValueType *>((reinterpret_cast<uint8_t *>(dev->data())
                                                              + b * dev->pitchBytes(0) + y * dev->pitchBytes(1)
                                                              + x * dev->pitchBytes(2) + k * dev->pitchBytes(3))));
                }
            }
        }
    }
}

// clang-format off
NVCV_TYPED_TEST_SUITE(
    Tensor4DWrapTensorTest, ttype::Types<
    ttype::Types<int, ttype::Value<NVCV_PIXEL_TYPE_S32>>,
    ttype::Types<uchar1, ttype::Value<NVCV_PIXEL_TYPE_U8>>,
    ttype::Types<const short2, ttype::Value<NVCV_PIXEL_TYPE_2S16>>,
    ttype::Types<uchar3, ttype::Value<NVCV_PIXEL_TYPE_3U8>>,
    ttype::Types<const uchar4, ttype::Value<NVCV_PIXEL_TYPE_4U8>>,
    ttype::Types<float3, ttype::Value<NVCV_PIXEL_TYPE_3F32>>,
    ttype::Types<const float4, ttype::Value<NVCV_PIXEL_TYPE_4F32>>
>);

// clang-format on

TYPED_TEST(Tensor4DWrapTensorTest, correct_with_tensor)
{
    using ValueType = ttype::GetType<TypeParam, 0>;
    auto pixelType  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Tensor tensor(
        {
            {3, 213, 211, 4},
            "NHWC"
    },
        nv::cv::PixelType{pixelType});

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor4DWrap<ValueType> wrap(*dev);

    auto pitchBytes = wrap.pitchBytes();
    EXPECT_EQ(pitchBytes[0], dev->pitchBytes(0));
    EXPECT_EQ(pitchBytes[1], dev->pitchBytes(1));
    EXPECT_EQ(pitchBytes[2], dev->pitchBytes(2));

    const ValueType *ptr0 = reinterpret_cast<const ValueType *>(dev->data());
    const ValueType *ptr1
        = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->data()) + dev->pitchBytes(0));
    const ValueType *ptr12 = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->data())
                                                                 + dev->pitchBytes(0) + 2 * dev->pitchBytes(1));
    const ValueType *ptr123
        = reinterpret_cast<const ValueType *>(reinterpret_cast<const uint8_t *>(dev->data()) + dev->pitchBytes(0)
                                              + 2 * dev->pitchBytes(1) + 3 * dev->pitchBytes(2));

    EXPECT_EQ(wrap.ptr(0), ptr0);
    EXPECT_EQ(wrap.ptr(1), ptr1);
    EXPECT_EQ(wrap.ptr(1, 2), ptr12);
    EXPECT_EQ(wrap.ptr(1, 2, 3), ptr123);
}

TYPED_TEST(Tensor4DWrapTensorTest, it_works_in_device)
{
    using ValueType = std::remove_cv_t<ttype::GetType<TypeParam, 0>>;
    auto pixelType  = ttype::GetValue<TypeParam, 1>;

    nv::cv::Tensor tensor(
        {
            {3, 213, 211, 4},
            "NHWC"
    },
        nv::cv::PixelType{pixelType});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int batches  = tensor.shape()[0];
    int height   = tensor.shape()[1];
    int width    = tensor.shape()[2];
    int channels = tensor.shape()[3];

    const auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(tensor.exportData());
    ASSERT_NE(dev, nullptr);

    cuda::Tensor4DWrap<ValueType> wrap(*dev);

    DeviceSetOnes(wrap, {width, height, batches, channels}, stream);

    long4  pitches{dev->pitchBytes(0), dev->pitchBytes(1), dev->pitchBytes(2), dev->pitchBytes(3)};
    size_t sizeBytes = batches * pitches.x;

    std::vector<uint8_t> test(sizeBytes);
    std::vector<uint8_t> gold(sizeBytes);

    for (int b = 0; b < batches; b++)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < channels; k++)
                {
                    *reinterpret_cast<ValueType *>(&gold[b * pitches.x + i * pitches.y + j * pitches.z + k * pitches.w])
                        = cuda::SetAll<ValueType>(1);
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
