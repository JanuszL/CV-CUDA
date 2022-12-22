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

function get_gitlab_auth_header()
{
    if [ "$PRIVATE_TOKEN" ]; then
        echo -n "PRIVATE-TOKEN: $PRIVATE_TOKEN"
    elif [ "$JOB_TOKEN" ]; then
        echo -n "JOB-TOKEN: $JOB_TOKEN"
    else
        echo "JOB_TOKEN or PRIVATE_TOKEN envvar must be set"
        exit 1
    fi
}

function get_gitlab_url()
{
    local v4_url=${API_V4_URL:-"https://gitlab-master.nvidia.com/api/v4"}
    local idproj=${PROJECT_ID:-70491} # cvcuda project
    echo -n "$v4_url/projects/$idproj"
}

function get_gitlab_package_id()
{
    if [ $# != 2 ]; then
        echo "Usage: get_gitlab_package_id <name> <version>"
        return 1
    fi

    local pkg_name=$1
    local pkg_version=$2
    shift 2

    curl --fail -G --header "$(get_gitlab_auth_header)" "$(get_gitlab_url)/packages" \
       	    --data-urlencode "package_name=$pkg_name" \
            --data-urlencode "package_type=generic" | jq -r ".[] | select(.version==\"$pkg_version\") | .id"
}

function get_gitlab_package_files()
{
    if [ $# != 1 ]; then
        echo "Usage: get_gitlab_package_files <package id>"
        return 1
    fi

    local idpkg=$1
    shift 1

    local header

    if [ "$LIST_PACKAGE_FILES_TOKEN" ]; then
        # hack for gitlab 15.6 not supporting CI_JOB_TOKEN for listing package files
        header="PRIVATE-TOKEN: $LIST_PACKAGE_FILES_TOKEN"
    else
        header=$(get_gitlab_auth_header)
    fi

    curl --fail --header "$header" "$(get_gitlab_url)/packages/$idpkg/package_files" | jq -r ".[] | [.file_name, .file_sha256] | @sh" | tr -d "'"
}

function download_gitlab_package_file()
{
    if [ $# != 4 ]; then
        echo "Usage: download_gitlab_package_files <package name> <package version> <file name> <output fname>"
        return 1
    fi

    local name=$1
    local version=$2
    local file=$3
    local output=$4
    shift 4

    curl --fail --header "$(get_gitlab_auth_header)" --output "$output" "$(get_gitlab_url)/packages/generic/$name/$version/$file" | jq
}

function upload_gitlab_package_file()
{
    if [ $# != 3 ]; then
        echo "Usage: upload_gitlab_package_files <package name> <package version> <file name>"
        return 1
    fi

    local name=$1
    local version=$2
    local file=$3
    shift 3

    curl --fail --header "$(get_gitlab_auth_header)" --upload-file "$file" "$(get_gitlab_url)/packages/generic/$name/$version/$file?select=package_file" | jq
}
