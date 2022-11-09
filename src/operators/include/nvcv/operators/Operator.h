/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file Operator.h
 *
 * @brief Defines types and functions to handle operators
 */

#ifndef NVCV_OP_OPERATOR_H
#define NVCV_OP_OPERATOR_H

#include "detail/Export.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NVCVOperator *NVCVOperatorHandle;

NVCV_OP_PUBLIC void nvcvOperatorDestroy(NVCVOperatorHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_OPERATOR_H */
