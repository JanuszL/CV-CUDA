#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

IMAGE_URL_BASE='gitlab-master.nvidia.com:5005/cv/cvcuda'

# image versions must be upgraded whenever a breaking
# change is done, such as removing some package, or updating
# packaged versions that introduces incompatibilities.
VER_IMAGE=0

VER_CUDA=11.7.0
VER_UBUNTU=22.04
