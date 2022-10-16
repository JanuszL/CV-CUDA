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

ARG VER_TRT=?

FROM nvcr.io/nvidia/tensorrt:$VER_TRT-py3
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    build-essential software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends gcc-11 g++-11 \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
RUN update-alternatives --set gcc /usr/bin/gcc-11
RUN update-alternatives --set g++ /usr/bin/g++-11

RUN apt-get update && apt-get install -y --no-install-recommends libgtest-dev libgmock-dev python3-pip \
    mlocate && updatedb \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip install torch torchvision
RUN echo "export PATH=$PATH:/opt/tensorrt/bin" >> ~/.bashrc
RUN echo $PATH
