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

if [[ $# != 1 && $# != 2 ]]; then
    echo "Create a stub DSO from a real DSO"
    echo "Usage: $(basename $0) <dso> [out_dso]"
    exit 1;
fi

dso=$1
stub_dso=$2

if [ ! -r "$dso" ]; then
    echo "$dso: file not found or is not readable"
    exit 1
fi

if [ -z "$stub_dso" ]; then
    tmp=$(basename "$dso")
    stub_dso="${tmp%.*}_stub.${tmp##*.}"
fi

CC=${CC:-gcc}
target=$($CC -v 2>&1 | sed -n 's@Target: @@pg')
#echo Targetting $target

if [ ${CC%-*} = $CC ]; then
    STRIP="strip"
else
    STRIP="$target-strip"
fi

tmp_c=$(mktemp).c
tmp_v=$(mktemp).v
tmp_orig_symbols=$(mktemp)
tmp_stub_symbols=$(mktemp)

function cleanup()
{
    rm -f $tmp_c $tmp_v $tmp_orig_symbols $tmp_stub_symbols
}

trap 'cleanup' EXIT

cat > $tmp_c <<-EOF
__attribute__((visibility("hidden")))
void do_abort()
{
    printf("stub function call\n");
    abort();
}
EOF

declare -a ver_symbols=()

resolver_emitted=0

strong_ifunc_name=""

function print_symbol()
{
    local name=$1
    local bind=$2
    local vis=$3
    local symtype=$4

    case "$vis" in
    PROTECTED)
        echo -n '__attribute__((visibility("protected"))) ' >> $tmp_c
        ;;
    DEFAULT)
        ;;
    *)
        echo "Visibility not understood: '$vis'"
        exit 1
    esac

    if [[ "$symtype" = IFUNC && "$resolver_emitted" = 0 ]]; then
        echo -n "static void (*ifunc_resolver())() { return do_abort; }" >> $tmp_c
        resolver_emitted=1
    fi

    case "$bind" in
    WEAK)
        if [[ $symtype != "IFUNC" ]]; then
            echo -n '__attribute__((weak)) ' >> $tmp_c
        fi
        ;;
    GLOBAL)
        ;;
    *)
        echo "Binding not understood: '$vis'"
        exit 1
    esac

    case "$symtype" in
    IFUNC)
        if [[ $bind = "WEAK" ]]; then
            if [[ "$strong_ifunc_name" ]]; then
                echo "extern void $name() __attribute__((weak, alias(\"$strong_ifunc_name\")));" >> $tmp_c
            else
                echo "Can't have a weak indirect function when no strong indirect function was defined"
                exit 1
            fi
        else
            echo "extern void $name() __attribute__((ifunc(\"ifunc_resolver\")));" >> $tmp_c
            strong_ifunc_name=$name
        fi
        ;;
    FUNC)
        echo "void $name() {do_abort();}" >> $tmp_c
        ;;
    OBJECT)
        echo "void *$name=(void*)0;" >> $tmp_c
        ;;
    TLS)
        echo "__thread void *$name=(void*)0;" >> $tmp_c
        ;;
    NOTYPE)
        ;;
    *)
        echo "Symbol type not understood: '$symtype'"
        exit 1;
    esac
}

symbol_filter='$7 !~ /(Ndx|UND|ABS|^$)/ && $5 !~ /(UNIQUE|LOCAL)/ && ( $5 ~ /WEAK/ || $4 !~ /NOTYPE/ )'
symbol_parser='{ print gensub(/^([^@]+)(@*)(.*)$/,"<\\3> <\\2> <\\1>", 1, $8),$4,$5,$6 }'

function get_symbols()
{
    local dso=$1

    # IFUNC on x86_64-redhat-linux DSOs (Centos7)
    # aren't well understood x86_64-linux-gnu (Ubuntu, Gentoo...)
    readelf -W --dyn-syms "$dso" | sed 's/<OS specific>: 10/IFUNC/g'
}

first=1
versioned=0

declare -a weak_ifuncs

while read ver vertype name symtype bind vis; do
    if [ -z "$name" ]; then
        continue;
    fi

    ver=${ver:1:-1}
    name=${name:1:-1}
    vertype=${vertype:1:-1}

    if [ -n "$ver" ]; then
        cver="$vertype$ver"
        cname=${name}_$cver
        cname="${cname//[.@-]/_}"
        echo -n "__asm__(\".symver ${cname},$name$vertype$ver\"); " >> $tmp_c
    else
        cname=${name}_nover
        cname="${cname//[.@-]/_}"
        echo -n "__asm__(\".symver ${cname},$name@\"); " >> $tmp_c
    fi

    if [[ $bind == WEAK && $symtype == IFUNC ]]; then
        weak_ifuncs+=("$cname $bind $vis $symtype")
    else
        print_symbol "$cname" "$bind" "$vis" "$symtype"
    fi

    if [ -z "$ver" ]; then
        ver_symbols+=("$cname")
        continue;
    fi

    if [[ -n $first || "$ver" != "$prev_ver" ]]; then
        if [[ -n "$prev_ver" ]]; then
            if [[ ${#ver_symbols[@]} -ne 0 ]]; then
                echo -e "local:" >> $tmp_v
                for f in "${ver_symbols[@]}"; do
                    echo "  $f;" >> $tmp_v
                done
                unset ver_symbols
            fi
            echo -e "};\n" >> $tmp_v
        fi

        echo -e "$ver {\nglobal:" >> $tmp_v
        first=""
    fi

    echo "  $name;" >> $tmp_v
    versioned=1

    ver_symbols+=("$cname")

    prev_ver=$ver
done <<< "$(get_symbols $dso | gawk "$symbol_filter $symbol_parser" | sort -V)"

# weak indirect functions must be defined last
for syminfo in "${weak_ifuncs[@]}"; do
    print_symbol $syminfo
done

if [ ! -f $tmp_c ]; then
    echo "No symbols found to be stubbed"
    exit 1
fi

if [ $versioned -eq 0 ]; then
    echo "{" >> $tmp_v
fi

if [[ ${#ver_symbols[@]} -ne 0 ]]; then
    echo -e "local:" >> $tmp_v
    for f in "${ver_symbols[@]}"; do
        echo "  $f;" >> $tmp_v
    done
fi

echo -e "};" >> $tmp_v
args="-Wl,--version-script=$tmp_v"

soname=$(readelf -d "$dso"  | gawk '/SONAME/ { print $5 }')
soname=${soname:1:-1}

if [ -n "$soname" ]; then
    args="$args -Wl,-soname=$soname"
fi

# -Wl,--no-ld-generated-unwind-info could help, but isn't supported on ld-2.30 (ubuntu 18.04)
# -Wl,--strip-all ends up breaking ifunc attribute
if ! $CC -o $stub_dso $tmp_c -Os -Wl,-no-eh-frame-hdr,-z,noseparate-code,-z,norelro -fPIC -fno-asynchronous-unwind-tables -Qn -Wno-implicit-function-declaration -fno-builtin -nostdlib -shared $args; then
    echo "Can't process $dso"
    exit 1
fi

rm -f $tmp_v $tmp_c

$STRIP -s $stub_dso

function list_contents()
{
    local parser='{ print $4,$5,$6,$8 }'
    local dso=$1
    get_symbols $dso | gawk "$symbol_filter $parser" | sort
}

list_contents $dso > $tmp_orig_symbols
list_contents $stub_dso > $tmp_stub_symbols

if ! diff $tmp_orig_symbols $tmp_stub_symbols; then
    echo "Error, resulting stub doesn't correspond to input"
    rm $stub_dso
    exit 1
fi

rm -f $tmp_stub_symbols $tmp_orig_symbols
