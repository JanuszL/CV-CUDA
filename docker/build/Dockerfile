# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Docker image used to build CV-CUDA on linux-x64

ARG VER_CUDA=?
ARG VER_UBUNTU=?

FROM nvidia/cuda:$VER_CUDA-devel-ubuntu$VER_UBUNTU

# need to update and install in one go, or else installation might use
# stale data from server stored in docker cache, with packages that don't exist anymore.
RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get install -y --no-install-recommends \
        git git-lfs \
        g++-11 \
        cmake ninja-build \
        ccache \
        libgtest-dev \
        pre-commit shellcheck \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Configure ccache
RUN mkdir -p /cache
COPY ccache.conf /etc/ccache.conf
ENV CCACHE_CONFIGPATH=/etc/ccache.conf
ENV PRE_COMMIT_HOME=/cache/pre-commit

# Allow using this image in systems without proper CUDA runtime/driver support.
# We'll be using this image only for building, don't need strict CUDA checks.
ENV NVIDIA_DISABLE_REQUIRE=true

#Documentation dependencies
WORKDIR /usr/cvcuda-tools/
RUN git clone https://gitlab-master.nvidia.com/omniverse/repo/repo_minimal.git
WORKDIR /usr/cvcuda/tools/repo_minimal
ADD repo-deps.packman.xml /usr/cvcuda-tools/repo_minimal/deps/
ENV PATH=$PATH:/usr/cvcuda-tools/repo_minimal
RUN chmod -R a+x /usr/cvcuda-tools/repo_minimal/repo
ADD repo.toml /usr/cvcuda-tools/repo_minimal/
RUN repo -h
