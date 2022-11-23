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

#include <common/TypedTests.hpp>
#include <nvcv/Config.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>

namespace t     = ::testing;
namespace ttest = nv::cv::test::type;
namespace nvcv  = nv::cv;

template<class T>
std::unique_ptr<T> CreateObj()
{
    if constexpr (std::is_same_v<nvcv::IImage, T>)
    {
        return std::make_unique<nvcv::Image>(nvcv::Size2D{64, 32}, nvcv::FMT_RGBA8);
    }
    else if constexpr (std::is_same_v<nvcv::IImageBatch, T>)
    {
        return std::make_unique<nvcv::ImageBatchVarShape>(32, nvcv::FMT_RGBA8);
    }
    else if constexpr (std::is_same_v<nvcv::IAllocator, T>)
    {
        return std::make_unique<nvcv::CustomAllocator<>>();
    }
    else if constexpr (std::is_same_v<nvcv::ITensor, T>)
    {
        return std::make_unique<nvcv::Tensor>(nvcv::TensorShape({32, 12, 4}, nvcv::TensorLayout::NONE), nvcv::TYPE_U8);
    }
    else
    {
        static_assert(sizeof(T) != 0 && "Invalid core object type");
    }
}

template<class T>
void SetMaxCount(int32_t maxCount)
{
    if constexpr (std::is_same_v<nvcv::IImage, T>)
    {
        nvcv::cfg::SetMaxImageCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::IImageBatch, T>)
    {
        nvcv::cfg::SetMaxImageBatchCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::IAllocator, T>)
    {
        nvcv::cfg::SetMaxAllocatorCount(maxCount);
    }
    else if constexpr (std::is_same_v<nvcv::ITensor, T>)
    {
        nvcv::cfg::SetMaxTensorCount(maxCount);
    }
    else
    {
        static_assert(sizeof(T) != 0 && "Invalid core object type");
    }
}

using AllCoreTypes = ttest::Types<nvcv::IImage, nvcv::IImageBatch, nvcv::ITensor, nvcv::IAllocator>;
NVCV_TYPED_TEST_SUITE(ConfigTests, AllCoreTypes);

TYPED_TEST(ConfigTests, set_max_obj_count_works)
{
    std::vector<std::unique_ptr<TypeParam>> objs;

    ASSERT_NO_THROW(SetMaxCount<TypeParam>(5));

    for (int i = 0; i < 5; ++i)
    {
        ASSERT_NO_THROW(objs.emplace_back(CreateObj<TypeParam>()));
    }

    NVCV_ASSERT_STATUS(NVCV_ERROR_OUT_OF_MEMORY, CreateObj<TypeParam>());

    objs.pop_back();
    ASSERT_NO_THROW(objs.emplace_back(CreateObj<TypeParam>()));
    NVCV_ASSERT_STATUS(NVCV_ERROR_OUT_OF_MEMORY, CreateObj<TypeParam>());
}

TYPED_TEST(ConfigTests, cant_change_limits_when_objects_are_alive)
{
    std::unique_ptr<TypeParam> obj = CreateObj<TypeParam>();

    NVCV_ASSERT_STATUS(NVCV_ERROR_INVALID_OPERATION, SetMaxCount<TypeParam>(5));

    obj.reset();

    ASSERT_NO_THROW(SetMaxCount<TypeParam>(5));
}
