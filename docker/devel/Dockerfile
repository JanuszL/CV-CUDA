# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Docker image used for development of CV-CUDA on linux-x64

ARG BASE_IMAGE=?
ARG TAG_IMAGE=?

FROM $BASE_IMAGE:$TAG_IMAGE

# need to update and install in one go, or else installation might use
# stale data from server stored in docker cache, with packages that don't exist anymore.
RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get install -y --no-install-recommends \
         vim \
         gdb cgdb \
         less \
         wget curl \
    && rm -rf /var/lib/apt/lists/*

# Enable CUDA driver checks as this image will be used for running CUDA programs
ENV NVIDIA_DISABLE_REQUIRE=false

# Config files we use
COPY vimrc /root/.vimrc
COPY gdbinit /root/.gdbinit

WORKDIR /cvcuda
