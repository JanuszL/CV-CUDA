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

#include <nvcv/Fwd.hpp>

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

    auto *devdata = dynamic_cast<const nvcv::IImageDataPitchDevice *>(data);
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

    const nvcv::IImageData *data = img.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::IImageDataPitchDevice *>(data);
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

TEST(ImageWrapData, wip_create)
{
    nvcv::ImageDataPitchDevice::Buffer buf;
    buf.numPlanes            = 1;
    buf.planes[0].width      = 173;
    buf.planes[0].height     = 79;
    buf.planes[0].pitchBytes = 190;
    buf.planes[0].buffer     = reinterpret_cast<void *>(678);

    nvcv::ImageWrapData img{
        nvcv::ImageDataPitchDevice{nvcv::FMT_U8, buf}
    };

    EXPECT_EQ(nvcv::Size2D(173, 79), img.size());
    EXPECT_EQ(nvcv::FMT_U8, img.format());
    ASSERT_NE(nullptr, img.handle());

    NVCVTypeImage type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetType(img.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGE_WRAPDATA, type);

    const nvcv::IImageData *data = img.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::IImageDataPitchDevice *>(data);
    ASSERT_NE(nullptr, devdata);

    ASSERT_EQ(1, devdata->numPlanes());
    EXPECT_EQ(img.format(), devdata->format());
    EXPECT_EQ(img.size(), devdata->size());
    EXPECT_EQ(img.size().w, devdata->plane(0).width);
    EXPECT_EQ(img.size().h, devdata->plane(0).height);
    EXPECT_LE(190, devdata->plane(0).pitchBytes);
    EXPECT_EQ(buf.planes[0].buffer, devdata->plane(0).buffer);
}

TEST(Image, wip_operator)
{
    namespace nvcv = nv::cv;

    nvcv::Image in{
        {512, 256},
        nvcv::FMT_RGBA8
    };
    nvcv::Image out{
        {512, 256},
        nvcv::FMT_RGBA8
    };

    auto *inData  = dynamic_cast<const nvcv::IImageDataPitchDevice *>(in.exportData());
    auto *outData = dynamic_cast<const nvcv::IImageDataPitchDevice *>(out.exportData());

    if (inData == nullptr || outData == nullptr)
    {
        throw std::runtime_error("Input and output images must have device-accessible pitch-linear memory");
    }
    if (inData->format() != outData->format())
    {
        throw std::runtime_error("Input and output images must have same format");
    }
    if (inData->size() != outData->size())
    {
        throw std::runtime_error("Input and output images must have same size");
    }

    assert(inData->numPlanes() == outData->numPlanes());

    for (int p = 0; p < inData->numPlanes(); ++p)
    {
        const nvcv::ImagePlanePitch &inPlane  = inData->plane(p);
        const nvcv::ImagePlanePitch &outPlane = outData->plane(p);

        cudaMemcpy2D(outPlane.buffer, outPlane.pitchBytes, inPlane.buffer, inPlane.pitchBytes,
                     (inData->format().planeBitsPerPixel(p) + 7) / 8 * inPlane.width, inPlane.height,
                     cudaMemcpyDeviceToDevice);
    }
}

TEST(ImageWrapData, wip_cleanup)
{
    nvcv::ImageDataPitchDevice::Buffer buf;
    buf.numPlanes            = 1;
    buf.planes[0].width      = 173;
    buf.planes[0].height     = 79;
    buf.planes[0].pitchBytes = 190;
    buf.planes[0].buffer     = reinterpret_cast<void *>(678);

    int  cleanupCalled = 0;
    auto cleanup       = [&cleanupCalled](const nvcv::IImageData &data)
    {
        ++cleanupCalled;
    };

    {
        nvcv::ImageWrapData img(nvcv::ImageDataPitchDevice{nvcv::FMT_U8, buf}, cleanup);
        EXPECT_EQ(0, cleanupCalled);
    }
    EXPECT_EQ(1, cleanupCalled) << "Cleanup must have been called when img got destroyed";
}

TEST(ImageWrapData, wip_mem_reqs)
{
    nvcv::Image::Requirements reqs = nvcv::Image::CalcRequirements({512, 256}, nvcv::FMT_NV12);

    nvcv::Image img(reqs);

    EXPECT_EQ(512, img.size().w);
    EXPECT_EQ(256, img.size().h);
    EXPECT_EQ(nvcv::FMT_NV12, img.format());

    const auto *data = dynamic_cast<const nvcv::IImageDataPitchDevice *>(img.exportData());

    ASSERT_NE(nullptr, data);
    ASSERT_EQ(2, data->numPlanes());
    EXPECT_EQ(512, data->plane(0).width);
    EXPECT_EQ(256, data->plane(0).height);

    EXPECT_EQ(256, data->plane(1).width);
    EXPECT_EQ(128, data->plane(1).height);

    EXPECT_EQ(data->plane(1).buffer,
              reinterpret_cast<std::byte *>(data->plane(0).buffer) + data->plane(0).pitchBytes * 256);

    for (int p = 0; p < 2; ++p)
    {
        EXPECT_EQ(cudaSuccess,
                  cudaMemset2D(data->plane(p).buffer, data->plane(p).pitchBytes, 123,
                               data->plane(p).width * img.format().planePixelStrideBytes(p), data->plane(p).height))
            << "Plane " << p;
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
