#!/bin/bash -e

# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Builds documentation based on sphinx and doxygen using repo_docs tool
# Usage: build_docs.sh [index_root path] [docs_dst folder]
# Ensure index_root path includes the index.rst file
# docs_dst folder is the folder where all the api documentation and html files are generated

if [[ -z "$1" || -z "$2" ]]; then
    echo "Usage: build_docs.sh [index_root path] [docs_dst folder]"
else
    # Used by repo.toml file
    export CVCUDA_DOCS_SRC_ROOT=$1
    export CVCUDA_DOCS_DST_ROOT=$2
    # Generate all docs
    /usr/cvcuda-tools/repo_minimal/repo docs
fi
