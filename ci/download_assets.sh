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

if [ $# != 2 ]; then
    echo "Usage: (opt)CACHEDIR=... (JOB_TOKEN=... or PRIVATE_TOKEN=...) $(basename $0) name version"
    exit 1
fi

PKG_NAME=$1
PKG_VERSION=$2
shift 2

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

. $SDIR/gitlab_utils.sh

CACHEDIR=${CACHEDIR:-/cache/gitlab_assets}
CACHEDIR=$CACHEDIR/$PKG_NAME/$PKG_VERSION
mkdir -p "$CACHEDIR"

idpkg=$(get_gitlab_package_id "$PKG_NAME" "$PKG_VERSION")

get_gitlab_package_files "$idpkg" | while read fname sha256; do
    if [ $sha256 != "$(sha256sum "$CACHEDIR/$fname" | cut -d' ' -f1)" ]; then
        tmpname=$(mktemp -p "$CACHEDIR")
        download_gitlab_package_file "$PKG_NAME" "$PKG_VERSION" "$fname" "$tmpname"
        echo "$sha256 $tmpname" | sha256sum -c --status
        mv -f $tmpname $CACHEDIR/$fname
    fi
done
