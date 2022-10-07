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

#include <common/ObjectBag.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/alloc/Allocator.h>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <thread>

namespace t    = ::testing;
namespace test = nv::cv::test;
namespace nvcv = nv::cv;

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_default)
{
    nvcv::CustomAllocator myalloc;

    void *ptrDev        = myalloc.deviceMem().alloc(768, 256);
    void *ptrHost       = myalloc.hostMem().alloc(160, 16);
    void *ptrHostPinned = myalloc.hostPinnedMem().alloc(144, 16);

    myalloc.deviceMem().free(ptrDev, 768, 256);
    myalloc.hostMem().free(ptrHost, 160, 16);
    myalloc.hostPinnedMem().free(ptrHostPinned, 144, 16);
}

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_custom_functors)
{
    int devCounter        = 1;
    int hostCounter       = 1;
    int hostPinnedCounter = 1;

    // clang-format off
    nvcv::CustomAllocator myalloc1
    {
        nvcv::CustomHostMemAllocator
        {
            [&hostCounter](int64_t size, int32_t align)
            {
                void *ptr = reinterpret_cast<void *>(hostCounter);
                hostCounter += size;
                return ptr;
            },
            [&hostCounter](void *ptr, int64_t size, int32_t align)
            {
                hostCounter -= size;
                assert(hostCounter == reinterpret_cast<ptrdiff_t>(ptr));
            }
        },
        nvcv::CustomDeviceMemAllocator
        {
            [&devCounter](int64_t size, int32_t align)
            {
                void *ptr = reinterpret_cast<void *>(devCounter);
                devCounter += size;
                return ptr;
            },
            [&devCounter](void *ptr, int64_t size, int32_t align)
            {
                devCounter -= size;
                assert(devCounter == reinterpret_cast<ptrdiff_t>(ptr));
            }
        },
        nvcv::CustomHostPinnedMemAllocator
        {
            [&hostPinnedCounter](int64_t size, int32_t align)
            {
                void *ptr = reinterpret_cast<void *>(hostPinnedCounter);
                hostPinnedCounter += size;
                return ptr;
            },
            [&hostPinnedCounter](void *ptr, int64_t size, int32_t align)
            {
                hostPinnedCounter -= size;
                assert(hostPinnedCounter == reinterpret_cast<ptrdiff_t>(ptr));
            }
        },
    };
    // clang-format on

    ASSERT_EQ((void *)1, myalloc1.hostMem().alloc(5));
    EXPECT_EQ(6, hostCounter);

    ASSERT_EQ((void *)1, myalloc1.hostPinnedMem().alloc(10));
    EXPECT_EQ(11, hostPinnedCounter);

    ASSERT_EQ((void *)1, myalloc1.deviceMem().alloc(7));
    EXPECT_EQ(8, devCounter);

    ASSERT_EQ((void *)8, myalloc1.deviceMem().alloc(2));
    EXPECT_EQ(10, devCounter);

    myalloc1.deviceMem().free((void *)8, 2);
    EXPECT_EQ(8, devCounter);

    myalloc1.deviceMem().free((void *)1, 7);
    EXPECT_EQ(1, devCounter);
}

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_custom_object)
{
    class MyDeviceAlloc : public nvcv::IDeviceMemAllocator
    {
    private:
        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            cudaMalloc(&ptr, size);
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            cudaFree(ptr);
        }
    };

    class MyHostAlloc : public nvcv::IHostMemAllocator
    {
    private:
        void *doAlloc(int64_t size, int32_t align) override
        {
            return ::malloc(size);
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            ::free(ptr);
        }
    };

    nvcv::CustomAllocator myalloc1{MyHostAlloc{}, MyDeviceAlloc{}};
}

TEST(Allocator, wip_test_custom_object_functor)
{
    class MyDeviceAlloc
    {
    public:
        void *alloc(int64_t size, int32_t align)
        {
            void *ptr;
            cudaMalloc(&ptr, size);
            return ptr;
        }

        void dealloc(void *ptr, int64_t size, int32_t align) noexcept
        {
            cudaFree(ptr);
        }
    };

    class MyHostAlloc
    {
    public:
        void *alloc(int64_t size, int32_t align)
        {
            return ::malloc(size);
        }

        void dealloc(void *ptr, int64_t size, int32_t align) noexcept
        {
            ::free(ptr);
        }
    };

    auto myDeviceAlloc = std::make_shared<MyDeviceAlloc>();
    auto myHostAlloc   = std::make_shared<MyHostAlloc>();

    // clang-format off
    nvcv::CustomAllocator myalloc1
    {
        nvcv::CustomHostMemAllocator{
            [myHostAlloc](int64_t size, int32_t align)
            {
                return myHostAlloc->alloc(size, align);
            },
            [myHostAlloc](void *ptr, int64_t size, int32_t align)
            {
                return myHostAlloc->dealloc(ptr, size, align);
            }
        },
        nvcv::CustomDeviceMemAllocator{
            [myDeviceAlloc](int64_t size, int32_t align)
            {
                return myDeviceAlloc->alloc(size, align);
            },
            [myDeviceAlloc](void *ptr, int64_t size, int32_t align)
            {
                return myDeviceAlloc->dealloc(ptr, size, align);
            }
        },
    };
    // clang-format on
}

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_custom_object_ref)
{
    class MyHostAlloc : public nvcv::IHostMemAllocator
    {
    private:
        void *doAlloc(int64_t size, int32_t align) override
        {
            return ::malloc(size);
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            ::free(ptr);
        }
    };

    MyHostAlloc myHostAlloc;

    nvcv::CustomAllocator myalloc1{
        std::ref(myHostAlloc),
    };

    nvcv::CustomAllocator myalloc2{std::ref(myHostAlloc)};

    auto myalloc3 = nvcv::CreateCustomAllocator(std::ref(myHostAlloc));

    EXPECT_EQ(&myHostAlloc, dynamic_cast<MyHostAlloc *>(&myalloc3.hostMem()));
    EXPECT_EQ(nullptr, dynamic_cast<MyHostAlloc *>(&myalloc3.deviceMem()));
}

class MyAsyncAlloc : public nvcv::IDeviceMemAllocator
{
public:
    void setStream(cudaStream_t stream)
    {
        m_stream = stream;
    }

private:
    void *doAlloc(int64_t size, int32_t align) override
    {
        void *ptr;
        EXPECT_EQ(cudaSuccess, cudaMallocAsync(&ptr, size, m_stream));
        return ptr;
    }

    void doFree(void *ptr, int64_t size, int32_t align) noexcept override
    {
        EXPECT_EQ(cudaSuccess, cudaFreeAsync(ptr, m_stream));
    }

    cudaStream_t m_stream = 0;
};

TEST(Allocator, wip_test_dali_stream_async)
{
    cudaStream_t stream1, stream2;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream1));
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream2));

    auto fn = [](cudaStream_t stream)
    {
        thread_local MyAsyncAlloc          myAsyncAlloc;
        thread_local nvcv::CustomAllocator myalloc{std::ref(myAsyncAlloc)};

        myAsyncAlloc.setStream(stream);

        void *ptr = myalloc.deviceMem().alloc(123, 5);
        myalloc.deviceMem().free(ptr, 123, 5);
    };

    std::thread thread1(fn, stream1);
    std::thread thread2(fn, stream2);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream1));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream2));

    thread1.join();
    thread2.join();

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

TEST(Allocator, wip_double_destroy_noop)
{
    NVCVAllocatorStorage allocStorage;
    NVCVAllocatorHandle  handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvAllocatorConstructCustom(nullptr, 0, &allocStorage, &handle));

    nvcvAllocatorDestroy(handle);

    void *ptr;
    NVCV_ASSERT_STATUS(NVCV_ERROR_INVALID_ARGUMENT, nvcvAllocatorFreeHostMemory(handle, &ptr, 16, 16));

    nvcvAllocatorDestroy(handle); // no-op, already destroyed
}

// disabled temporary while the API isn't stable
#if 0

// Parameter space validity tests ============================================

/********************************************
 *         nvcvMemAllocatorCreate
 *******************************************/

class MemAllocatorCreateParamTest
    : public t::TestWithParam<std::tuple<test::Param<"handle", bool>,        // 0
                                         test::Param<"numCustomAllocs", int> // 1
                                             NVCVStatus>>                    // 2
{
public:
    MemAllocatorCreateParamTest()
        : m_goldStatus(std::get<1>(GetParam()))
    {
        if (std::get<0>(GetParam()))
        {
            EXPECT_EQ(NVCV_SUCCESS, nvcvMemAllocatorCreate(&m_paramHandle));
        }
        else
        {
            m_paramHandle = nullptr;
        }
    }

    ~MemAllocatorCreateParamTest()
    {
        nvcvMemAllocatorDestroy(m_paramHandle);
    }

protected:
    NVCVMemAllocator m_paramHandle;
    NVCVStatus       m_goldStatus;
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive_handle, MemAllocatorCreateParamTest,
                              test::Value(true) * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_handle, MemAllocatorCreateParamTest,
                              test::Value(false) * NVCV_ERROR_INVALID_ARGUMENT);

// clang-format on

TEST_P(MemAllocatorCreateParamTest, stream)
{
    NVCVMemAllocator handle = nullptr;
    EXPECT_EQ(m_goldStatus, nvcvMemAllocatorCreate(m_paramHandle ? &handle : nullptr));

    nvcvMemAllocatorDestroy(handle); // to avoid memleaks
}

/********************************************
 *         nvcvMemAllocatorDestroy
 *******************************************/

using MemAllocatorDestroyParamTest = MemAllocatorCreateParamTest;

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive_handle, MemAllocatorDestroyParamTest,
                              test::ValueList{true, false} * NVCV_SUCCESS);

// clang-format on

TEST_P(MemAllocatorDestroyParamTest, stream)
{
    // Must not crash or assert, but we can't test that without using
    // googletest's Death Tests, as it involves forking the process.
    nvcvMemAllocatorDestroy(m_paramHandle);
    m_paramHandle = nullptr;
}

/********************************************
 *    nvcvMemAllocatorSetCustomAllocator
 *******************************************/

class MemAllocatorSetAllocatorParamTest
    : public t::TestWithParam<std::tuple<test::Param<"handle", bool, true>,                     // 0
                                         test::Param<"memtype", NVCVMemoryType, NVCV_MEM_HOST>, // 1
                                         test::Param<"fnMalloc", bool, true>,                   // 2
                                         test::Param<"fnFree", bool, true>,                     // 3
                                         test::Param<"ctx", bool, false>,                       // 4
                                         NVCVStatus>>                                           // 5
{
public:
    MemAllocatorSetAllocatorParamTest()
        : m_paramMemType(std::get<1>(GetParam()))
        , m_goldStatus(std::get<5>(GetParam()))
    {
        if (std::get<0>(GetParam()))
        {
            EXPECT_EQ(NVCV_SUCCESS, nvcvMemAllocatorCreate(&m_paramHandle));
        }

        if (std::get<2>(GetParam()))
        {
            // Dummy implementations
            static auto fnMalloc = [](void *, int64_t, int32_t, uint32_t) -> void *
            {
                return nullptr;
            };
            m_paramFnAllocMem = fnMalloc;
        }

        if (std::get<3>(GetParam()))
        {
            static auto fnFree = [](void *, void *, int64_t, int32_t, uint32_t) -> void {
            };
            m_paramFnFreeMem = fnFree;
        }

        if (std::get<4>(GetParam()))
        {
            m_paramContext = this;
        }
    }

    ~MemAllocatorSetAllocatorParamTest()
    {
        nvcvMemAllocatorDestroy(m_paramHandle);
    }

protected:
    NVCVMemAllocator m_paramHandle     = nullptr;
    NVCVMemAllocFunc m_paramFnAllocMem = nullptr;
    ;
    NVCVMemFreeFunc m_paramFnFreeMem = nullptr;
    void           *m_paramContext   = nullptr;
    NVCVMemoryType  m_paramMemType;
    NVCVStatus      m_goldStatus;
};

static test::ValueList g_ValidMemTypes = {NVCV_MEM_HOST, NVCV_MEM_CUDA, NVCV_MEM_CUDA_PINNED};

static test::ValueList g_InvalidMemTypes = {
    (NVCVMemoryType)-1,
    (NVCVMemoryType)NVCV_NUM_MEMORY_TYPES,
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive_handle, MemAllocatorSetAllocatorParamTest,
                              test::Value(true) * Dup<4>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_handle, MemAllocatorSetAllocatorParamTest,
                              test::Value(false) * Dup<4>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_memtype, MemAllocatorSetAllocatorParamTest,
                              Dup<1>(test::ValueDefault()) * g_ValidMemTypes * Dup<3>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_memtype, MemAllocatorSetAllocatorParamTest,
                              Dup<1>(test::ValueDefault()) * g_InvalidMemTypes * Dup<3>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_fnMalloc, MemAllocatorSetAllocatorParamTest,
                              Dup<2>(test::ValueDefault()) * test::Value(true) * Dup<2>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_fnMalloc, MemAllocatorSetAllocatorParamTest,
                              Dup<2>(test::ValueDefault()) * test::Value(false) * Dup<2>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_fnFree, MemAllocatorSetAllocatorParamTest,
                              Dup<3>(test::ValueDefault()) * test::Value(true) * Dup<1>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_fnFree, MemAllocatorSetAllocatorParamTest,
                              Dup<3>(test::ValueDefault()) * test::Value(false) * Dup<1>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_context, MemAllocatorSetAllocatorParamTest,
                              Dup<4>(test::ValueDefault()) * test::ValueList{true,false}
                              * NVCV_SUCCESS);

// clang-format on

TEST_P(MemAllocatorSetAllocatorParamTest, test)
{
    EXPECT_EQ(m_goldStatus, nvcvMemAllocatorSetCustomAllocator(m_paramHandle, m_paramMemType, m_paramFnAllocMem,
                                                               m_paramFnFreeMem, m_paramContext));
}

// Execution tests ===========================================

class MemAllocatorCreateExecTest : public t::Test
{
protected:
    test::ObjectBag m_bag;
};

TEST_F(MemAllocatorCreateExecTest, handle_filled_in)
{
    NVCVMemAllocator handle = nullptr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMemAllocatorCreate(&handle));
    m_bag.insert(handle);

    EXPECT_NE(nullptr, handle);
}

#endif
