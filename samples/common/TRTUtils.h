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
 * @file TRTUtils.h
 *
 * @brief TensorRT Utilities
 */

#ifndef NVCV_TRTBACKEND_H
#define NVCV_TRTBACKEND_H

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Struct to store TensorRT blob dimensions.
 */
struct TRTBackendBlobSize
{
    int channels; /**< channels count. */
    int height;   /**< blob height. */
    int width;    /**< blob width. */
};

/**
 * TensorRT wrapper class.
 */
class TRTBackend
{
public:
    /**
     * Constructor of TRTBackend.
     * @param modelFilePath path of the network model.
     * @param precision TensorRT precision type.
     */
    TRTBackend(const char *modelFilePath, int batchSize = 1);

    /**
     * Destructor of TRTBackend.
     */
    ~TRTBackend();

    /**
     * Run inference.
     * @param buffer input GPU buffers.
     * @param batchSize run infer with specific batch size, passed in setBindingDimension() call.
     * @param stream update cuda stream in this instance.
     */
    void infer(void **buffer, int batchSize, cudaStream_t stream);

    /**
     * Get all input/output bindings count.
     * @return number of all bindings.
     */
    int getBlobCount() const;

    /**
     * Get the blob dimension for given blob index.
     * @param blobIndex blob index.
     * @return blob dimension for the given index.
     */
    TRTBackendBlobSize getTRTBackendBlobSize(int blobIndex) const;

    /**
     * Get the total number of elements for the given blob index.
     * @param blobIndex blob index.
     * @return total size for the given index.
     */
    int getBlobLinearSize(int blobIndex) const;

    /**
     * Get the blob index for the given blob name.
     * @param blobName blob name.
     * @return blob index for the given name.
     */
    int getBlobIndex(const char *blobName) const;

    /**
     * Check if binding is input.
     * @param index binding index.
     * @return whether the binding is input.
     */
    bool bindingIsInput(const int index) const;

    /**
     * Get Layer Name
     * @param index binding index.
     * @return Binding Layer Name
     */
    const char *getLayerName(const int index) const;

private:
    // Forward declaration
    struct TRTImpl;
    // TRT related variables
    std::unique_ptr<TRTImpl> m_pImpl;
};

#endif // NVCV_TRTBACKEND_H
