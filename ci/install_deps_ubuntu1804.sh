#!/bin/bash -eE

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

if [ -z "$CUDA_VERSION" ]; then
    echo "CUDA_VERSION envvar must be set"
fi

cuda_version=$(echo $CUDA_VERSION | sed 's@\([0-9]\+\)\.\([0-9]\+\).*@\1-\2@g')
cuda_major=$(echo $CUDA_VERSION | sed 's@\([0-9]\+\).*@\1@g')

apt-get update

# Basic packages needed to fetch packages from gitlab (not part of README)
apt-get install -y --no-install-recommends curl jq
export JOB_TOKEN=${JOB_TOKEN:-$CI_JOB_TOKEN}
export PROJECT_ID=${PROJECT_ID:-$CI_PROJECT_ID}
export API_V4_URL=${API_V4_URL:-$CI_API_V4_URL}
export CACHEDIR=/cache/gitlab_assets/

# 1. Set up cv-cuda repo
apt-get install -y --no-install-recommends git git-lfs
# Ubuntu 18.04 doesn't have pre-commit, we can skip it

# 2. Build CV-CUDA
# 2.1 set up for g++-11
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository --yes ppa:ubuntu-toolchain-r/test

# 2.2 set up for cmake>=3.22
apt-get install -y --no-install-recommends build-essential libssl-dev
curl https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

apt-get update

apt-get install -y --no-install-recommends g++-11 cmake ninja-build python3-dev libssl-dev
apt-get install -y --no-install-recommends cuda-minimal-build-$cuda_version

# 3. Build documentation
apt-get install -y --no-install-recommends doxygen python3 python3-pip
python3 -m pip install sphinx-rtd-theme sphinx==4.5.0 breathe exhale recommonmark graphviz sphinx-rtd-theme

# 4. Build Samples
apt-get install -y --no-install-recommends python3 python3-pip

$SDIR/download_assets.sh torch 1.13.0
python3 -m pip install $CACHEDIR/torch/1.13.0/*.whl

$SDIR/download_assets.sh torchvision 0.14.0
python3 -m pip install $CACHEDIR/torchvision/0.14.0/*.whl

$SDIR/download_assets.sh tensorrt 8.5.2.2-cuda$cuda_major
mkdir -p /opt/tensorrt
tar --strip-components=1 -C /opt/tensorrt -xf $CACHEDIR/tensorrt/8.5.2.2-cuda$cuda_major/TensorRT*.tar*
ln -sf /opt/tensorrt/bin/* /usr/bin/
ln -sf /opt/tensorrt/include/* /usr/include/
ln -sf /opt/tensorrt/lib/* /usr/lib/x86_64-linux-gnu/

python3 -m pip install /opt/tensorrt/python/tensorrt-*cp310*.whl

apt-get install -y --no-install-recommends libnvjpeg-dev-$cuda_version libcudnn8

# 5. Build Tests
python3.10 -m pip install pytest numba torch
