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

function cleanup()
{
    rm -f stub_symbols orig_symbols libtest.so libtest_stub.so
}

trap 'cleanup' EXIT

function run_test()
{
    local name=$1
    local src=$2
    local ver=$3

    echo "test $name"

    local args=''
    if [ -n "$ver" ]; then
        args="-Wl,--version-script=$ver"
    fi

    gcc -o libtest.so -shared -fPIC $args -Wl,-soname=libtest.so -xc - <<<$src
    rm -f libtest_stub.so

    ./mkstub.sh libtest.so
    function list_contents()
    {
        readelf --dyn-syms "$1" | awk '$7 !~ /(UND|^$|Ndx)/ { print $4,$5,$6,$8 }' | sort
    }

    list_contents libtest.so > orig_symbols
    list_contents libtest_stub.so > stub_symbols

    diff orig_symbols stub_symbols
}

run_test versioned "$(cat test_versioned.c)" test_versioned.v
run_test mixed "$(cat test_versioned.c test_noversion.c)" test_versioned.v
run_test "not versioned" "$(cat test_noversion.c)"
