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

#include "OpSIFT.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace legacy = nvcv::legacy::cuda_op;

namespace cvcuda::priv {

SIFT::SIFT(int3 maxShape, int maxOctaveLayers)
{
    //TODO init
}

SIFT::~SIFT()
{
    //TODO de-init
}

void SIFT::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &featCoords,
                      const nvcv::Tensor &featMetadata, const nvcv::Tensor &featDescriptors,
                      const nvcv::Tensor &numFeatures, int numOctaveLayers, float contrastThreshold,
                      float edgeThreshold, float initSigma, NVCVSIFTFlagType flags) const
{
    //TODO add calls to kernel here
}

} // namespace cvcuda::priv
