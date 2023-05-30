/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OpMinMaxLoc.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

namespace {

// Utilities for min/max operator ----------------------------------------------

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

using TensorDataRef         = std::reference_wrapper<nvcv::TensorDataStridedCuda>;
using OptionalTensorData    = nvcv::Optional<nvcv::TensorDataStridedCuda>;
using OptionalTensorDataRef = nvcv::Optional<std::reference_wrapper<nvcv::TensorDataStridedCuda>>;

// Run functions in layers -----------------------------------------------------

// The 3rd run layer is after template instantiation ---------------------------

template<typename T, class DataStridedCuda>
inline void RunMinMaxLocForType(cudaStream_t stream, const DataStridedCuda &inData, OptionalTensorDataRef minValData,
                                OptionalTensorDataRef minLocData, OptionalTensorDataRef numMinData,
                                OptionalTensorDataRef maxValData, OptionalTensorDataRef maxLocData,
                                OptionalTensorDataRef numMaxData)
{
    // run CUDA kernels here
}

// The 2nd run layer is after exporting output data ----------------------------

template<class DataStridedCuda>
inline void RunMinMaxLocDataOut(cudaStream_t stream, const DataStridedCuda &inData, nvcv::DataType inDataType,
                                OptionalTensorDataRef minValData, OptionalTensorDataRef minLocData,
                                OptionalTensorDataRef numMinData, OptionalTensorDataRef maxValData,
                                OptionalTensorDataRef maxLocData, OptionalTensorDataRef numMaxData)
{
    switch (inDataType)
    {
#define NVCV_CASE_MINMAXLOC(DT, T)                                                                         \
    case nvcv::TYPE_##DT:                                                                                  \
        RunMinMaxLocForType<T>(stream, inData, minValData, minLocData, numMinData, maxValData, maxLocData, \
                               numMaxData);                                                                \
        break

        NVCV_CASE_MINMAXLOC(U8, uchar1);
        NVCV_CASE_MINMAXLOC(U16, ushort1);
        NVCV_CASE_MINMAXLOC(U32, uint1);
        NVCV_CASE_MINMAXLOC(S8, char1);
        NVCV_CASE_MINMAXLOC(S16, short1);
        NVCV_CASE_MINMAXLOC(S32, int1);
        NVCV_CASE_MINMAXLOC(F32, float1);
        NVCV_CASE_MINMAXLOC(F64, double1);

#undef NVCV_CASE_MINMAXLOC

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type");
    }
}

// This is used in the 1st run layer to checks if input data type matches the min/max value data type

inline bool DataTypeMatches(nvcv::DataType inDataType, nvcv::DataType valDataType)
{
    bool match = false;
    switch (valDataType)
    {
    case nvcv::TYPE_S32:
        match = inDataType == nvcv::TYPE_S32 || inDataType == nvcv::TYPE_S16 || inDataType == nvcv::TYPE_S8;
        break;

    case nvcv::TYPE_U32:
        match = inDataType == nvcv::TYPE_U32 || inDataType == nvcv::TYPE_U16 || inDataType == nvcv::TYPE_U8;
        break;

    case nvcv::TYPE_F32:
    case nvcv::TYPE_F64:
        match = inDataType == valDataType;
        break;

    default:
        break;
    }
    return match;
}

// The 1st run layer is after exporting input data -----------------------------

template<class DataStridedCuda>
inline void RunMinMaxLocDataIn(cudaStream_t stream, const DataStridedCuda &inData, nvcv::DataType inDataType,
                               int inNumSamples, int inNumChannels, const nvcv::Tensor &minVal,
                               const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                               const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax)
{
    if (inNumSamples == 0)
    {
        return;
    }
    if (inNumSamples < 0 && inNumSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of samples in the input %d must be in [0, 65535]", inNumSamples);
    }
    if (inNumChannels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have a single channel, not %d",
                              inNumChannels);
    }

    if ((minVal && (!minLoc || !numMin)) || (minLoc && (!minVal || !numMin)) || (numMin && (!minVal || !minLoc)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output minVal, minLoc and numMin must be provided together");
    }
    if ((maxVal && (!maxLoc || !numMax)) || (maxLoc && (!maxVal || !numMax)) || (numMax && (!maxVal || !maxLoc)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output maxVal, maxLoc and numMax must be provided together");
    }
    if (!minVal && !maxVal)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "At least one output (minVal and/or maxVal) must be chosen");
    }

    OptionalTensorData    minValData, minLocData, numMinData, maxValData, maxLocData, numMaxData;
    OptionalTensorDataRef minValRef, minLocRef, numMinRef, maxValRef, maxLocRef, numMaxRef;

    if (minVal)
    {
        minValData = minVal.exportData<nvcv::TensorDataStridedCuda>();
        if (!minValData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minVal must be cuda-accessible, pitch-linear tensor");
        }
        minLocData = minLoc.exportData<nvcv::TensorDataStridedCuda>();
        if (!minLocData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minLoc must be cuda-accessible, pitch-linear tensor");
        }
        numMinData = numMin.exportData<nvcv::TensorDataStridedCuda>();
        if (!numMinData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMin must be cuda-accessible, pitch-linear tensor");
        }

        minValRef = TensorDataRef(*minValData);
        minLocRef = TensorDataRef(*minLocData);
        numMinRef = TensorDataRef(*numMinData);

        if (!DataTypeMatches(inDataType, minValData->dtype()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Wrong output minVal data type %s for input tensor data type %s output minVal data "
                                  "type must be S32/U32/F32/F64: for input data type S8/S16/S32 use S32; for "
                                  "U8/U16/U32 use U32; for all other data types use same data type as input tensor",
                                  nvcvDataTypeGetName(minValData->dtype()), nvcvDataTypeGetName(inDataType));
        }
        if (!((minValData->rank() == 0 && inNumSamples == 1)
              || ((minValData->rank() == 1 || minValData->rank() == 2) && inNumSamples == minValData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minVal number of samples must be the same as input tensor");
        }
        if (minValData->rank() == 2 && minValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minVal number of channels must be 1, not %ld", minValData->shape(1));
        }
        if (!((minLocData->rank() == 1 && inNumSamples == 1)
              || (minLocData->rank() == 2 && inNumSamples == minLocData->shape(0))
              || (minLocData->rank() == 3 && inNumSamples == minLocData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minLoc number of samples must be the same as input tensor");
        }
        if (!((numMinData->rank() == 0 && inNumSamples == 1)
              || ((numMinData->rank() == 1 || numMinData->rank() == 2) && inNumSamples == numMinData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMin number of samples must be the same as input tensor");
        }
        if (numMinData->rank() == 2 && minValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMin number of channels must be 1, not %ld", numMinData->shape(1));
        }
        if (!((minLocData->rank() == 3 && minLocData->dtype() == nvcv::TYPE_S32 && minLocData->shape(2) == 2)
              || (minLocData->rank() == 3 && minLocData->dtype() == nvcv::TYPE_2S32 && minLocData->shape(2) == 1)
              || (minLocData->rank() == 2 && minLocData->dtype() == nvcv::TYPE_2S32)))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output minLoc must have rank 2 or 3 and 2xS32 or 2S32 data type, "
                                  "not rank %d and data type %s",
                                  minLocData->rank(), nvcvDataTypeGetName(minLocData->dtype()));
        }
        if (numMinData->dtype() != nvcv::TYPE_S32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output numMin must have S32 data type, not %s",
                                  nvcvDataTypeGetName(numMinData->dtype()));
        }
    }

    if (maxVal)
    {
        maxValData = maxVal.exportData<nvcv::TensorDataStridedCuda>();
        if (!maxValData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxVal must be cuda-accessible, pitch-linear tensor");
        }
        maxLocData = maxLoc.exportData<nvcv::TensorDataStridedCuda>();
        if (!maxLocData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxLoc must be cuda-accessible, pitch-linear tensor");
        }
        numMaxData = numMax.exportData<nvcv::TensorDataStridedCuda>();
        if (!numMaxData)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMax must be cuda-accessible, pitch-linear tensor");
        }

        maxValRef = TensorDataRef(*maxValData);
        maxLocRef = TensorDataRef(*maxLocData);
        numMaxRef = TensorDataRef(*numMaxData);

        if (!DataTypeMatches(inDataType, maxValData->dtype()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Wrong output maxVal data type %s for input tensor data type %s output maxVal data "
                                  "type must be S32/U32/F32/F64: for input data type S8/S16/S32 use S32; for "
                                  "U8/U16/U32 use U32; for all other data types use same data type as input tensor",
                                  nvcvDataTypeGetName(maxValData->dtype()), nvcvDataTypeGetName(inDataType));
        }
        if (!((maxValData->rank() == 0 && inNumSamples == 1)
              || ((maxValData->rank() == 1 || maxValData->rank() == 2) && inNumSamples == maxValData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxVal number of samples must be the same as input tensor");
        }
        if (maxValData->rank() == 2 && maxValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxVal number of channels must be 1, not %ld", maxValData->shape(1));
        }
        if (!((maxLocData->rank() == 1 && inNumSamples == 1)
              || (maxLocData->rank() == 2 && inNumSamples == maxLocData->shape(0))
              || (maxLocData->rank() == 3 && inNumSamples == maxLocData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxLoc number of samples must be the same as input tensor");
        }
        if (!((numMaxData->rank() == 0 && inNumSamples == 1)
              || ((numMaxData->rank() == 1 || numMaxData->rank() == 2) && inNumSamples == numMaxData->shape(0))))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMax number of samples must be the same as input tensor");
        }
        if (numMaxData->rank() == 2 && maxValData->shape(1) != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output numMax number of channels must be 1, not %ld", numMaxData->shape(1));
        }
        if (!((maxLocData->rank() == 3 && maxLocData->dtype() == nvcv::TYPE_S32 && maxLocData->shape(2) == 2)
              || (maxLocData->rank() == 3 && maxLocData->dtype() == nvcv::TYPE_2S32 && maxLocData->shape(2) == 1)
              || (maxLocData->rank() == 2 && maxLocData->dtype() == nvcv::TYPE_2S32)))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Output maxLoc must have rank 2 or 3 and 2xS32 or 2S32 data type, "
                                  "not rank %d and data type %s",
                                  maxLocData->rank(), nvcvDataTypeGetName(maxLocData->dtype()));
        }
        if (numMaxData->dtype() != nvcv::TYPE_S32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output numMax must have S32 data type, not %s",
                                  nvcvDataTypeGetName(numMaxData->dtype()));
        }
    }

    RunMinMaxLocDataOut(stream, inData, inDataType, minValRef, minLocRef, numMinRef, maxValRef, maxLocRef, numMaxRef);
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

MinMaxLoc::MinMaxLoc() {}

// Tensor operator -------------------------------------------------------------

void MinMaxLoc::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &minVal,
                           const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                           const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);

    RunMinMaxLocDataIn(stream, *inData, inData->dtype(), inAccess->numSamples(), inAccess->numChannels(), minVal,
                       minLoc, numMin, maxVal, maxLoc, numMax);
}

// VarShape operator -----------------------------------------------------------

void MinMaxLoc::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &minVal,
                           const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                           const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax) const
{
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    nvcv::ImageFormat inFormat = inData->uniqueFormat();

    if (inFormat.numPlanes() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Image batches must have a single plane, not %d",
                              inFormat.numPlanes());
    }

    RunMinMaxLocDataIn(stream, *inData, inFormat.planeDataType(0), inData->numImages(), inFormat.planeNumChannels(0),
                       minVal, minLoc, numMin, maxVal, maxLoc, numMax);
}

} // namespace cvcuda::priv
