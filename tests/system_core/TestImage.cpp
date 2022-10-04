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

#include <nvcv/Image.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

namespace nvcv = nv::cv;

TEST(Image, wip_create)
{
    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8);

    EXPECT_EQ(nvcv::Size2D(163, 117), img.size());
    EXPECT_EQ(nvcv::FMT_RGBA8, img.format());
    ASSERT_NE(nullptr, img.handle());

    NVCVTypeImage type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetType(img.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGE, type);

    const nvcv::IImageData *data = img.exportData();
    ASSERT_NE(nullptr, data);

    EXPECT_NE(nullptr, dynamic_cast<nvcv::AllocatorWrapHandle *>(&img.alloc()));

    auto *devdata = dynamic_cast<const nvcv::IImageDataDevicePitch *>(data);
    ASSERT_NE(nullptr, devdata);

    ASSERT_EQ(1, devdata->numPlanes());
    EXPECT_EQ(img.format(), devdata->format());
    EXPECT_EQ(img.size(), devdata->size());
    EXPECT_EQ(img.size().w, devdata->plane(0).width);
    EXPECT_EQ(img.size().h, devdata->plane(0).height);
    EXPECT_LE(163 * 4, devdata->plane(0).pitchBytes);
    EXPECT_NE(nullptr, devdata->plane(0).buffer);

    const nvcv::ImagePlanePitch &plane = devdata->plane(0);

    EXPECT_EQ(cudaSuccess, cudaMemset2D(plane.buffer, plane.pitchBytes, 123, plane.width * 4, plane.height));
}

TEST(Image, wip_create_managed)
{
    namespace nvcv = nv::cv;

    // clang-format off
    nvcv::CustomAllocator managedAlloc
    {
        nvcv::CustomDeviceMemAllocator
        {
            [](int64_t size, int32_t)
            {
                 void *ptr = nullptr;
                 cudaMallocManaged(&ptr, size);
                 return ptr;
            },
            [](void *ptr, int64_t, int32_t)
            {
                cudaFree(ptr);
            }
        }
    };
    // clang-format on

    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8, &managedAlloc);

    EXPECT_EQ(&managedAlloc, &img.alloc());

    const nvcv::IImageData *data = img.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::IImageDataDevicePitch *>(data);
    ASSERT_NE(nullptr, devdata);

    ASSERT_EQ(1, devdata->numPlanes());
    EXPECT_LE(163 * 4, devdata->plane(0).pitchBytes);
    EXPECT_NE(nullptr, devdata->plane(0).buffer);

    const nvcv::ImagePlanePitch &plane = devdata->plane(0);

    EXPECT_EQ(cudaSuccess, cudaMemset2D(plane.buffer, plane.pitchBytes, 123, plane.width * 4, plane.height));

    for (int i = 0; i < plane.height; ++i)
    {
        std::byte *beg = reinterpret_cast<std::byte *>(plane.buffer) + plane.pitchBytes * i;
        std::byte *end = beg + plane.width * 4;

        ASSERT_EQ(end, std::find_if(beg, end, [](std::byte b) { return b != std::byte{123}; }))
            << "All bytes in the image must be 123";
    }
}

// Future API ideas
#if 0
TEST(Image, wip_image_managed_memory)
{
    namespace nvcv = nv::cv;

    nvcv::CustomAllocator managedAlloc
    {
        nvcv::CustomDeviceMemAllocator
        {
            [](int64_t size, int32_t)
            {
                void *ptr = nullptr;
                cudaMallocManaged(&ptr, size);
                return ptr;
            },
            [](void *ptr, int64_t, int32_t)
            {
                cudaFree(ptr);
            }
        }
    };

    nvcv::Image img({512, 256}, nvcv::FMT_RGBA8, &managedAlloc);

    EXPECT_EQ(nvcv::Size2D{512,256}, img.size());
    EXPECT_EQ(nvcv::FMT_RGBA8, img.format());

    {
        nvcv::LockImageData lkData = img.lock(nvcv::READ);
        if(auto *data = dynamic_cast<const nvcv::IImageDataDeviceMem *>(lkData->data()))
        {
            cv::GpuMat ocvGPU{data->size.h, data->size.w,
                              data->plane(0).buffer,
                              data->plane(0).pitchBytes};
            // ...

        }
        else if(auto *data = dynamic_cast<const nvcv::IImageDataCudaArray *>(lkData->data()))
        {
            cudaArray_t carray = data->plane(0);
            // ...
        }
    }

    if(nvcv::LockImageData lkData = img.lock<nvcv::IImageDataDeviceMem>(nvcv::READ))
    {
        cv::GpuMat ocvGPU{lkData->size.h, lkData->size.w,
                          lkData->plane(0).buffer,
                          lkData->plane(0).pitchBytes};
        // ...
    }

    // alternative?
    if(nvcv::LockImageData lkData = img.lockDeviceMem(nvcv::READ))
    {
        // If we know image holds managed memory, we can do this:
        cv::Mat ocvCPU{lkData->size.h, lkData->size.w,
                       lkData->plane(0).buffer,
                       lkData->plane(0).pitchBytes};
        // ...
    }

    class ProcessImageVisitor
        : public nvcv::IVisitorImageData
    {
    public:
        ProcessImageVisitor(cudaStream_t stream)
            : m_stream(stream) {}

        bool visit(IImageDataDeviceMem &data) override
        {
            // pitch-linear processing
            cv::GpuMat ocvGPU{data->size.h, data->size.w,
                              data->plane(0).buffer,
                              data->plane(0).pitchBytes};
            // process image in m_stream
            return true;
        }

        bool visit(IImageDataCudaArray &data) override
        {
            // block-linear processing
            cudaArray_t carray = data->plane(0);
            // process image in m_stream
            return true;
        }

    }

    // Works for both pitch-linear and block-linear
    img.lock(nvcv::READ).accept(ProcessImageVisitor(stream));
}

TEST(Image, wip_wrap_opencv_read)
{
    namespace nvcv = nv::cv;

    // create opencv mat and wrap it
    cv::Mat mat(256,512,CV_8UC3);
    nvcv::ImageWrapData img(mat, nvcv::FMT_BGR8);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lock(nvcv::LOCK_READ);
        cv::imwrite("output.png",mat);
    }
}

TEST(Image, wip_wrap_opencv_write)
{
    namespace nvcv = nv::cv;

    // create opencv mat and wrap it
    cv::Mat mat(256,512,CV_8UC3);
    nvcv::ImageWrapData img(mat, nvcv::FMT_BGR8);

    {
        nvcv::LockedImage lk = img.lock(nvcv::LOCK_WRITE);
        // write to mat
    }

    // ... op read from img ...
}

TEST(Image, wip_img_opencv_read)
{
    namespace nvcv = nv::cv;

    nvcv::Image img({512, 256}, nvcv::FMT_BGR8, nvcv::HOST);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lockOpenCV(nvcv::LOCK_READ); // - dev->host copy
        cv::imwrite("output.png",*lk);
    }
}

TEST(Image, wip_memcpy_opencv_read)
{
    namespace nvcv = nv::cv;

    nvcv::ImageWrapData img({512,256}, nvcv::FMT_BGR8);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lockDevice(nvcv::LOCK_READ);

        cv::Mat mat(256,512,CV_8UC3);
        memcpy(mat, *lk);

        cv::imwrite("output.png",mat);
    }
}


TEST(Image, wip_memcpy_opencv_write)
{
    namespace nvcv = nv::cv;

    nvcv::ImageWrapData img({512,256}, nvcv::FMT_BGR8);

    // read result from disk
    {
        nvcv::LockedImage lk = img.lockDevice(nvcv::LOCK_READ);

        cv::Mat mat = cv::imread("input.png");

        memcpy(*lk, mat);
    }

    // ... op write to img ...
}

#endif
