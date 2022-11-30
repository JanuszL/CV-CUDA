#!/bin/bash -e

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

# Given an executable (or dso) A and a list of DSOs B, find the symbols
# that missing in A and try to find equivalents (maybe from older version) in B.

# It outputs a list of missing symbol in A that could and could not be found in B.

if [ $# -lt 1 ]; then
    echo "Invalid args. Usage: $(basename $0) <target exec> [old lib,...]"
    exit 1
fi

target=$1
shift

oldlibs="$*"

missingsyms="$(ldd -r $target 2>&1 | awk '/^symbol.*/ { print $2 }' | sed s/,//g | sort | uniq)"

misspattern="^("
for s in $missingsyms; do
    misspattern+="|$s"
done
misspattern+=")@@"

total_found="" # symbol name + version tag

for lib in $oldlibs; do
    libsyms="$(readelf -sW $lib | awk '$4 ~ /FUNC/ && $5 ~ /(GLOBAL|WEAK)/ && $6 ~ /DEFAULT/ && $7 !~ /UND/ && $8 ~ /.*@@/ { print $8 }' | sort | uniq)"
    found="$(echo "$libsyms" | egrep "$misspattern" | sed 's/@@/@/g' || true)"
    echo "------ $lib" 1>&2
    if [ "$found" ]; then
        #if [ "$total_found" ]; then
        #    total_found+="\n"
        #fi
        total_found+="\n$found"
    fi
done

total_found="$(echo -e "$total_found" | egrep -v "^[[:space:]]*$" | sort | uniq)"

notfound="$(diff <( echo "$missingsyms" ) <( echo "$total_found" | awk -F '@' '{ print $1 }') || true)"

if [ "$notfound" ]; then
    echo -e "\nSymbols not found:"
    echo "$notfound" | awk '/^</ { print $2 }'
fi

if [ "$total_found" ]; then
    echo -e "\nSymbols to be added:"
    echo -e "$total_found"
fi
