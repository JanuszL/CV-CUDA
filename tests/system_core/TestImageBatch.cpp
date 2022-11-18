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

#include <common/ValueTests.hpp>
#include <nvcv/ImageBatch.hpp>

#include <list>
#include <random>

namespace nvcv = nv::cv;
namespace t    = ::testing;
namespace test = nv::cv::test;

static bool operator==(const NVCVImagePlanePitch &a, const NVCVImagePlanePitch &b)
{
    return a.width == b.width && a.height == b.height && a.pitchBytes == b.pitchBytes && a.buffer == b.buffer;
}

static std::ostream &operator<<(std::ostream &out, const NVCVImagePlanePitch &a)
{
    return out << a.width << 'x' << a.height << '@' << a.pitchBytes << ':' << a.buffer;
}

TEST(ImageBatchVarShape, wip_create)
{
    nvcv::ImageBatchVarShape batch(100, nvcv::FMT_NV12);

    EXPECT_EQ(100, batch.capacity());
    EXPECT_EQ(nvcv::FMT_NV12, batch.format());
    EXPECT_EQ(0, batch.numImages());
    ASSERT_NE(nullptr, batch.handle());

    NVCVTypeImageBatch type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchGetType(batch.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGEBATCH_VARSHAPE, type);

    EXPECT_NE(nullptr, dynamic_cast<nvcv::AllocatorWrapHandle *>(&batch.alloc()));

    // empty data
    {
        const nvcv::IImageBatchData *data = batch.exportData(0);
        ASSERT_NE(nullptr, data);

        ASSERT_EQ(batch.format(), data->format());

        auto *devdata = dynamic_cast<const nvcv::IImageBatchVarShapeDataPitchDevice *>(data);
        ASSERT_NE(nullptr, devdata);

        ASSERT_EQ(0, devdata->numImages());
        EXPECT_NE(nullptr, devdata->imgPlanes());

        EXPECT_EQ(nvcv::Size2D(0, 0), devdata->maxSize());
    }

    std::vector<NVCVImagePlanePitch> goldPlanes;
    std::vector<NVCVImageHandle>     goldHandles;

    auto addToGold = [&goldPlanes, &goldHandles](const nvcv::IImage &img)
    {
        auto *imgdata = dynamic_cast<const nvcv::IImageDataPitchDevice *>(img.exportData());
        EXPECT_NE(nullptr, imgdata);
        if (imgdata)
        {
            for (int i = 0; i < imgdata->numPlanes(); ++i)
            {
                goldPlanes.push_back(imgdata->plane(i));
            }
            goldHandles.push_back(img.handle());
        }
    };

    auto calcMaxSize = [&goldPlanes]()
    {
        nvcv::Size2D maxSize = {0, 0};
        for (size_t i = 0; i < goldPlanes.size(); ++i)
        {
            maxSize.w = std::max(maxSize.w, goldPlanes[i].width);
            maxSize.h = std::max(maxSize.h, goldPlanes[i].height);
        }
        return maxSize;
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Add a bunch of images, using different methods

    nvcv::Image img0{
        {320, 128},
        nvcv::FMT_NV12
    };
    batch.pushBack(img0);
    addToGold(img0);

    nvcv::Image img1{
        {320, 128},
        nvcv::FMT_NV12
    };
    batch.pushBack(&img1, &img1 + 1);
    addToGold(img1);

    std::list<nvcv::Image> vec0;
    for (int i = 0; i < 10; ++i)
    {
        vec0.emplace_back(nvcv::Size2D{328 + i * 2, 130 - i * 2}, nvcv::FMT_NV12);
        addToGold(vec0.back());
    }
    batch.pushBack(vec0.begin(), vec0.end());

    std::vector<std::unique_ptr<nvcv::IImage>> vec1;
    for (int i = 0; i < 10; ++i)
    {
        vec1.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{328 + i * 2, 130 - i * 2}, nvcv::FMT_NV12));
        addToGold(*vec1.back());
    }
    batch.pushBack(vec1.begin(), vec1.end());

    // To synchronize buffers
    const nvcv::IImageBatchVarShapeData *vsdata = batch.exportData(stream); // test output type
    const auto *devdata = dynamic_cast<const nvcv::IImageBatchVarShapeDataPitchDevice *>(vsdata);
    ASSERT_NE(nullptr, devdata);
    EXPECT_EQ(calcMaxSize(), devdata->maxSize());

    std::vector<std::shared_ptr<nvcv::IImage>> vec2;
    for (int i = 0; i < 10; ++i)
    {
        vec2.emplace_back(std::make_shared<nvcv::Image>(nvcv::Size2D{328 + i * 2, 130 - i * 2}, nvcv::FMT_NV12));
        addToGold(*vec2.back());
    }
    batch.pushBack(vec2.begin(), vec2.end());

    std::vector<std::reference_wrapper<nvcv::IImage>> vec3;
    for (nvcv::Image &img : vec0)
    {
        vec3.emplace_back(img);
        addToGold(vec3.back().get());
    }
    batch.pushBack(vec3.begin(), vec3.end());

    // Remove some
    batch.popBack(5);
    goldPlanes.erase(goldPlanes.end() - 5 * 2, goldPlanes.end());
    goldHandles.erase(goldHandles.end() - 5, goldHandles.end());

    // To synchronize buffers
    devdata = dynamic_cast<const nvcv::IImageBatchVarShapeDataPitchDevice *>(batch.exportData(stream));
    ASSERT_NE(nullptr, devdata);
    EXPECT_EQ(calcMaxSize(), devdata->maxSize());

    // use callback
    std::vector<std::shared_ptr<nvcv::IImage>> vec4;
    batch.pushBack(
        [&]()
        {
            if (vec4.size() < 5)
            {
                int i = vec4.size();

                auto img = std::make_shared<nvcv::Image>(nvcv::Size2D{320 + i * 2, 122 - i * 2}, nvcv::FMT_NV12);
                addToGold(*img);
                vec4.push_back(img);
                return img;
            }
            else
            {
                return std::shared_ptr<nvcv::Image>{};
            }
        });

    // not-empty data
    {
        const nvcv::IImageBatchData *data = batch.exportData(stream);
        ASSERT_NE(nullptr, data);

        ASSERT_EQ(batch.format(), data->format());

        auto *devdata = dynamic_cast<const nvcv::IImageBatchVarShapeDataPitchDevice *>(data);
        ASSERT_NE(nullptr, devdata);

        ASSERT_EQ(goldHandles.size(), devdata->numImages());
        EXPECT_NE(nullptr, devdata->imgPlanes());

        EXPECT_EQ(calcMaxSize(), devdata->maxSize());

        std::vector<NVCVImagePlanePitch> planes(devdata->numImages() * 2);
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(planes.data(), devdata->imgPlanes(), sizeof(planes[0]) * planes.size(),
                                               cudaMemcpyDeviceToHost, stream));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        EXPECT_THAT(planes, t::ElementsAreArray(goldPlanes));

        int cur = 0;
        for (auto it = batch.begin(); it != batch.end(); ++it, ++cur)
        {
            EXPECT_EQ(goldHandles[cur], it->handle()) << "Image #" << cur;
        }
    }

    {
        nvcv::ImageBatchVarShapeWrapHandle wrap(batch.handle());
        EXPECT_EQ(batch.capacity(), wrap.capacity());
        EXPECT_EQ(batch.format(), wrap.format());
        ASSERT_EQ(batch.numImages(), wrap.numImages());
        EXPECT_EQ(batch.handle(), wrap.handle());

        int  cur    = 0;
        auto itwrap = wrap.begin();
        for (auto itgold = batch.begin(); itgold != batch.end(); ++itgold, ++itwrap, ++cur)
        {
            EXPECT_EQ(itgold->handle(), itwrap->handle()) << "Image #" << cur;
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(ImageBatchVarShape, wip_sync)
{
    std::vector<NVCVImagePlanePitch> goldPlanes;
    std::vector<NVCVImageHandle>     goldHandles;

    auto addToGold = [&goldPlanes, &goldHandles](const nvcv::IImage &img)
    {
        auto *imgdata = dynamic_cast<const nvcv::IImageDataPitchDevice *>(img.exportData());
        EXPECT_NE(nullptr, imgdata);
        if (imgdata)
        {
            goldPlanes.push_back(imgdata->plane(0));
            goldPlanes.push_back(imgdata->plane(1));
            goldHandles.push_back(img.handle());
        }
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape batch(10000, nvcv::FMT_NV12);

    std::mt19937                  rng(123);
    std::uniform_int_distribution rnd(1, 4);

    std::list<nvcv::Image> vec0;
    for (int i = 0; i < batch.capacity(); ++i)
    {
        vec0.emplace_back(nvcv::Size2D{rnd(rng) * 2, rnd(rng) * 2}, nvcv::FMT_NV12);
        addToGold(vec0.back());
    }

    std::list<nvcv::Image>       vec1;
    std::vector<NVCVImageHandle> vec1Handles;
    for (int i = 0; i < batch.capacity(); ++i)
    {
        vec1.emplace_back(nvcv::Size2D{rnd(rng) * 2, rnd(rng) * 2}, nvcv::FMT_NV12);
        vec1Handles.push_back(vec1.back().handle());
    }

    batch.pushBack(vec0.begin(), vec0.end());

    // trigger host->dev async copy
    auto *devdata = dynamic_cast<const nvcv::IImageBatchVarShapeDataPitchDevice *>(batch.exportData(stream));

    // Re-write batch contents in host-side, must have waited
    // until async copy finishes
    batch.clear();
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchVarShapePushImages(batch.handle(), vec1Handles.data(), vec1Handles.size()));

    // test device buffer against gold, new data from vec1 must not
    // show up
    {
        ASSERT_EQ(goldHandles.size(), devdata->numImages());
        EXPECT_NE(nullptr, devdata->imgPlanes());

        std::vector<NVCVImagePlanePitch> planes(devdata->numImages() * 2);
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(planes.data(), devdata->imgPlanes(), sizeof(planes[0]) * planes.size(),
                                               cudaMemcpyDeviceToHost, stream));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        EXPECT_THAT(planes, t::ElementsAreArray(goldPlanes));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
