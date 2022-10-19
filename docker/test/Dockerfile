# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Docker image used to test CV-CUDA on linux-x64

ARG VER_CUDA=?
ARG VER_UBUNTU=?

FROM nvidia/cuda:$VER_CUDA-runtime-ubuntu$VER_UBUNTU

ARG VER_CUDA=?

# For testing python bindings ======================================

# Add deadsnakes apt repo to fetch older python versions
ADD deadsnakes-ubuntu-ppa-jammy.list /etc/apt/sources.list.d
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776

# Add so that tzdata don't ask for timezone info in a noninteractive installation.
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.7 \
        python3.8 \
        python3.9 \
        python3.10 \
    && rm -rf /var/lib/apt/lists/*
