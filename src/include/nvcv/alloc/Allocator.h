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

/**
 * @file Allocator.h
 *
 * @brief Defines the public C interface to NVCV resource allocators.
 *
 * Allocators objects allow users to override the resource
 * allocation strategy used by several NVCV entities, such as images,
 * operators, etc.
 *
 * NVCV currently support three resource types:
 * - host memory: memory directly accessible by CPU.
 * - device memory : memory directly accessible by cuda-enabled GPU.
 * - host pinned memory: memory directly accessible by both CPU and cuda-enabled GPU.
 *
 * By default, the following functions are used to allocate
 * and deallocate resourcees for these types.
 *
 * @anchor default_res_allocators
 *
 * | Resource type      | Malloc        | Free         |
 * |--------------------|---------------|--------------|
 * | host memory        | malloc        | free         |
 * | device memory      | cudaMalloc    | cudaFree     |
 * | host pinned memory | cudaHostAlloc | cudaHostFree |
 *
 * By using defining custom resource allocators, user can override the allocation
 * and deallocation functions used for each resource type. When overriding, they can pass
 * a pointer to some user-defined context. It'll be passed unchanged to the
 * corresponding malloc and free function. This allows passing, for instance, a
 * pointer to an object whose methods will be called from inside the overriden
 * functions.
 */

#ifndef NVCV_ALLOCATOR_H
#define NVCV_ALLOCATOR_H

#include "../Status.h"
#include "../detail/Export.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** Function type for memory resource allocation.
 *
 * @param [in] ctx        Pointer to user context.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        + Must be >= 0.
 * @param [in] alignBytes Address alignment in bytes.
 *                        It's guaranteed to be a power of two.
 *                        The returned address will be multiple of this value.
 *
 * @returns Pointer to allocated memory buffer.
 *          Must return NULL if buffer cannot be allocated.
 */
typedef void *(*NVCVMemAllocFunc)(void *ctx, int64_t sizeBytes, int32_t alignBytes);

/** Function type for memory deallocation.
 *
 * @param [in] ctx        Pointer to user context.
 * @param [in] ptr        Pointer to memory buffer to be deallocated.
 *                        If NULL, the operation must do nothing, successfully.
 * @param [in] sizeBytes, alignBytes Parameters passed during buffer allocation.
 */
typedef void (*NVCVMemFreeFunc)(void *ctx, void *ptr, int64_t sizeBytes, int32_t alignBytes);

/** Memory types handled by the memory resource allocator. */
typedef enum
{
    NVCV_RESOURCE_MEM_HOST,       /**< Memory accessible by host (CPU). */
    NVCV_RESOURCE_MEM_DEVICE,     /**< Memory accessible by device (GPU). */
    NVCV_RESOURCE_MEM_HOST_PINNED /**< Memory accessible by both host and device. */
} NVCVResourceType;

#define NVCV_NUM_RESOURCE_TYPES (3)

typedef struct NVCVCustomMemAllocatorRec
{
    /** Pointer to function that performs memory allocation.
     *  + Function must return memory buffer with type specified by memType.
     *  + Cannot be NULL.
     */
    NVCVMemAllocFunc fnAlloc;

    /** Pointer to function that performs memory deallocation.
     *  + Function must deallocate memory allocated by @ref fnMemAlloc.
     *  + Cannot be NULL.
     */
    NVCVMemFreeFunc fnFree;
} NVCVCustomMemAllocator;

typedef union NVCVCustomResourceAllocatorRec
{
    NVCVCustomMemAllocator mem;
} NVCVCustomResourceAllocator;

typedef struct NVCVCustomAllocatorRec
{
    /** Pointer to user context.
     *  It's passed unchanged to memory allocation/deallocation functions.
     *  It can be NULL, in this case no context is passed in.
     */
    void *ctx;

    /** Type of memory being handled by fnMemAlloc and fnMemFree. */
    NVCVResourceType resType;

    NVCVCustomResourceAllocator res;
} NVCVCustomAllocator;

/** Handle to an allocator instance. */
typedef struct NVCVAllocatorImpl *NVCVAllocator;

/** Creates a custom allocator instance.
 *
 * The created allocator is configured to use the default resource
 * allocator functions specified @ref default_mem_allocators "here".
 *
 * When not needed anymore, the allocator instance must be destroyed by
 * @ref nvcvAllocatorDestroy function.
 *
 * @param [in] customAllocators    Array of custom resource allocators.
 *                                    + There must be at most one custom allocator for each memory type.
 *                                    + Restrictions on the custom allocator members apply,
 *                                      see \ref NVCVCustomAllocator.
 * @param [in] numCustomAllocators Number of custom allocators in the array.
 * @param [out] halloc Where new instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the allocator.
 * @retval #NVCV_SUCCESS                Allocator created successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorCreateCustom(const NVCVCustomAllocator *customAllocators,
                                                 int32_t numCustomAllocators, NVCVAllocator *halloc);

/** Destroys an existing allocator instance.
 *
 * @note All objects that depend on the allocator instance must already be destroyed,
 *       if not undefined behavior will ensue, possibly segfaults.
 *
 * @param [in] halloc Memory allocator to be destroyed.
 *                    If NULL, no operation is performed, successfully.
 *                    + The handle must have been created with @ref nvcvAllocatorCreate.
 */
NVCV_PUBLIC void nvcvAllocatorDestroy(NVCVAllocator halloc);

/** Allocates a memory buffer of a host-accessible memory.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the resource allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [out] ptr       Holds a pointer to the allocated buffer.
 *                        + Cannot be NULL.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        + Must be >= 0.
 * @param [in] alignBytes Address alignment in bytes.
 *                        The returned address will be multiple of this value.
 *                        + Must a power of 2.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough free memory.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorAllocHostMemory(NVCVAllocator halloc, void **ptr, int64_t sizeBytes,
                                                    int32_t alignBytes);

/** Frees a host-accessible memory buffer.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the memory allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [in] memType    Type of memory to be freed.
 * @param [in] ptr        Pointer to the memory buffer to be freed.
 *                        It can be NULL. In this case, no operation is performed.
 *                        + Must have been allocated by @ref nvcvAllocatorAllocHostMemory.
 * @param [in] sizeBytes,alignBytes Parameters passed during buffer allocation.
 *                                  + Not passing the exact same parameters
 *                                    passed during allocation will lead to undefined behavior.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorFreeHostMemory(NVCVAllocator halloc, void *ptr, int64_t sizeBytes,
                                                   int32_t alignBytes);

/** Allocates a memory buffer of both host- and device-accessible memory.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the resource allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [out] ptr       Holds a pointer to the allocated buffer.
 *                        + Cannot be NULL.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        + Must be >= 0.
 * @param [in] alignBytes Address alignment in bytes.
 *                        The returned address will be multiple of this value.
 *                        + Must a power of 2.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough free memory.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorAllocHostPinnedMemory(NVCVAllocator halloc, void **ptr, int64_t sizeBytes,
                                                          int32_t alignBytes);

/** Frees a both host- and device-accessible memory buffer.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the memory allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [in] memType    Type of memory to be freed.
 * @param [in] ptr        Pointer to the memory buffer to be freed.
 *                        It can be NULL. In this case, no operation is performed.
 *                        + Must have been allocated by @ref nvcvAllocatorAllocHostPinnedMemory.
 * @param [in] sizeBytes,alignBytes Parameters passed during buffer allocation.
 *                                  + Not passing the exact same parameters
 *                                    passed during allocation will lead to undefined behavior.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorFreeHostPinnedMemory(NVCVAllocator halloc, void *ptr, int64_t sizeBytes,
                                                         int32_t alignBytes);

/** Allocates a memory buffer of device-accessible memory.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the resource allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [out] ptr       Holds a pointer to the allocated buffer.
 *                        + Cannot be NULL.
 * @param [in] sizeBytes  How many bytes to allocate.
 *                        + Must be >= 0.
 * @param [in] alignBytes Address alignment in bytes.
 *                        The returned address will be multiple of this value.
 *                        + Must a power of 2.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough free memory.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorAllocDeviceMemory(NVCVAllocator halloc, void **ptr, int64_t sizeBytes,
                                                      int32_t alignBytes);

/** Frees a device-accessible memory buffer.
 *
 * It's usually used when implementing operators.
 *
 * @param [in] halloc     Handle to the memory allocator object to be used.
 *                        + Must have been created by @ref nvcvAllocatorCreate.
 * @param [in] memType    Type of memory to be freed.
 * @param [in] ptr        Pointer to the memory buffer to be freed.
 *                        It can be NULL. In this case, no operation is performed.
 *                        + Must have been allocated by @ref nvcvAllocatorAllocDeviceMemory.
 * @param [in] sizeBytes,alignBytes Parameters passed during buffer allocation.
 *                                  + Not passing the exact same parameters
 *                                    passed during allocation will lead to undefined behavior.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation completed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvAllocatorFreeDeviceMemory(NVCVAllocator halloc, void *ptr, int64_t sizeBytes,
                                                     int32_t alignBytes);

#ifdef __cplusplus
}
#endif

#endif // NVCV_ALLOCATOR_H
