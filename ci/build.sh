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

# Creates the build tree and builds CV-CUDA

# Usage: build.sh [build type] [build dir] [additional cmake args]
# build type:
#   - debug or release
#   - If not specified, defaults to release
# build dir:
#   - Where build tree will be created
#   - If not specified, defaults to either build-rel or build-deb, depending on the build type

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

# Command line parsing ===============================================

# Defaults
build_type="release"
build_dir=""
source_dir="$SDIR/.."

if [[ $# -ge 1 ]]; then
    case $1 in
    debug|release)
        build_type=$1
        if [[ $# -ge 2 ]]; then
            build_dir=$2
            shift
        fi
        shift
        ;;
    *)
        build_dir=$1
        shift
        ;;
    esac
fi

# Store additional cmake args user might have passed
user_args="$*"

# Create build directory =============================================

# If build dir not explicitely defined,
if [[ -z "$build_dir" ]]; then
    # Uses one derived from build type
    build_dir="build-${build_type:0:3}"
fi
mkdir -p "$build_dir"

# Set build configuration depending on build type ====================

# Common config
cmake_args="-DBUILD_TESTS=1"

if [[ "$ENABLE_PYTHON" = '0' || "$ENABLE_PYTHON" = 'no' ]]; then
    cmake_args="$cmake_args -DBUILD_PYTHON=0"
else
    # enables python by default or when asked
    cmake_args="$cmake_args -DBUILD_PYTHON=1"
fi

if [ "$PYTHON_VERSIONS" ]; then
    cmake_args="-DPYTHON_VERSIONS=$PYTHON_VERSIONS"
fi

case $build_type in
    release)
        cmake_args="$cmake_args -DCMAKE_BUILD_TYPE=Release"
        ;;
    debug)
        cmake_args="$cmake_args -DCMAKE_BUILD_TYPE=Debug"
        ;;
esac

# Configure build toolchain ===========================================

# Make sure we use most recent gcc-11.x
CC=$(find /usr/bin/gcc-11* | sort -rV | head -n 1)
CXX=$(find /usr/bin/g++-11* | sort -rV | head -n 1)
export CC CXX

# Prefer to use ninja if found
if which ninja > /dev/null; then
    cmake_args="$cmake_args -G Ninja"
    export NINJA_STATUS="[%f/%t %r %es] "
fi

# Config ccache
unset has_ccache
if which ccache > /dev/null; then
    has_ccache=1
fi
if [[ $has_ccache ]]; then
    ccache_stats=$(pwd)/$build_dir/ccache_stats.log
    rm -rf "$ccache_stats"
    cmake_args="${cmake_args} -DCCACHE_STATSLOG=${ccache_stats}"
fi

# config CUDA
CUDA_MAJOR=11
for nvcc_path in /usr/local/cuda-$CUDA_MAJOR/bin/nvcc /usr/local/cuda/bin/nvcc; do
    if [ -x "$nvcc_path" ]; then
        cmake_args="$cmake_args -DCMAKE_CUDA_COMPILER=$nvcc_path"
        break
    fi
done

# Create build tree and build! ===========================================

# Create build tree
cmake -B "$build_dir" "$source_dir"  \
    -DBUILD_TESTS=1 \
    $cmake_args \
    $user_args

# Build CV-CUDA
cmake --build "$build_dir" -- $MAKE_OPTS

# Show ccache status, if available!
if [[ $has_ccache ]]; then
    # Show build stats
    CCACHE_STATSLOG=${ccache_stats} ccache --show-stats -V
fi
