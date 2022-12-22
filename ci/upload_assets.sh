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

if [ $# -lt 3 ]; then
    echo "Usage: (opt)CACHEDIR=... (JOB_TOKEN=... or PRIVATE_TOKEN=...) $(basename $0) name version files..."
    exit 1
fi

PKG_NAME=$1
PKG_VERSION=$2
shift 2

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

. $SDIR/gitlab_utils.sh

CACHEDIR=${CACHEDIR:-/cache/gitlab_assets/$PKG_NAME/$PKG_VERSION}
mkdir -p "$CACHEDIR"

idpkg=$(get_gitlab_package_id "$PKG_NAME" "$PKG_VERSION")

declare -A server_files_sha256=()

while read fname sha256; do
    if [ "$fname" ]; then
        server_files_sha256["$fname"]="$sha256"
        echo "$fname - $sha256"
    fi
done <<< "$(get_gitlab_package_files "$idpkg")"

for file in "$@"; do
    echo "Processing $file to $PKG_NAME/$PKG_VERSION"

    fname=$(basename "$file")
    server_sha256=${server_files_sha256[$fname]}

    if [ -z "$server_sha256" ]; then
	echo "   upload started"
        upload_gitlab_package_file "$PKG_NAME" "$PKG_VERSION" "$file"
	echo "   upload finished"
    else
        file_sha256=$(sha256sum "$file" | cut -d' ' -f1)
	if [ "$server_sha256" == "$file_sha256" ]; then
            echo "  file already exist on server, skipping"
	elif [ "$server_sha256" ]; then
            echo "  file already exist but server sha256 $server_sha256 doesn't file's $file_sha256, skipping"
	fi
    fi
done
