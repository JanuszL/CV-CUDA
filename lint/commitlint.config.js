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

module.exports = {
    extends : [ "@commitlint/config-conventional" ],
    helpUrl : 'https://confluence.nvidia.com/display/CVCUDA/Commit+message+format',
    rules : {'references-empty' : [ 2, 'never' ]},
    parserPreset : {parserOpts : {issuePrefixes : [ '^CVCUDA-' ]}}
}
