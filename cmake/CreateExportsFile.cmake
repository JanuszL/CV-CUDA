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

if(NOT OUTPUT)
    message(FATAL_ERROR "No output exports file specified")
endif()

if(NOT SOURCES)
    message(FATAL_ERROR "No source files specified")
endif()

string(REPLACE " " ";" SOURCES ${SOURCES})

# Create an empty file
file(WRITE ${OUTPUT} "")

set(all_versions "")

foreach(src ${SOURCES})
    file(STRINGS ${SOURCE_DIR}/${src} funcdef_list REGEX "NVCV_DEFINE_API.*")

    foreach(func_def ${funcdef_list})
        if(func_def MATCHES "^NVCV_DEFINE_API\\(+([^,]+),([^,]+),[^,]+,([^,]+).*$")
            string(STRIP "${CMAKE_MATCH_1}" ver_major)
            string(STRIP "${CMAKE_MATCH_2}" ver_minor)
            string(STRIP "${CMAKE_MATCH_3}" func)
            list(APPEND all_versions ${ver_major}.${ver_minor})
            list(APPEND funcs_${ver_major}_${ver_minor} ${func})
        else()
            message(FATAL_ERROR "I don't understand ${func_def}")
        endif()
    endforeach()
endforeach()

list(SORT all_versions COMPARE NATURAL)
list(REMOVE_DUPLICATES all_versions)

if(all_versions)
    set(prev_version "")
    foreach(ver ${all_versions})
        if(ver MATCHES "([0-9]+)\\.([0-9]+)")
            set(ver_major ${CMAKE_MATCH_1})
            set(ver_minor ${CMAKE_MATCH_2})

            file(APPEND ${OUTPUT} "NVCV_${ver} {\nglobal:\n")

            if(NOT funcs_${ver_major}_${ver_minor})
                message(FATAL_ERROR "funcs_${ver_major}_${ver_minor} must not be empty")
            endif()

            list(SORT funcs_${ver_major}_${ver_minor})

            foreach(func ${funcs_${ver_major}_${ver_minor}})
                file(APPEND ${OUTPUT} "    ${func};\n")
            endforeach()

            if(prev_version)
                file(APPEND ${OUTPUT} "} NVCV_${prev_version};\n\n")
            else()
                file(APPEND ${OUTPUT} "local: *;\n};\n\n")
            endif()

            set(prev_version ${ver})
        else()
            message(FATAL_ERROR "I don't version ${ver}")
        endif()
    endforeach()
else()
    file(APPEND ${OUTPUT} "NVCV {\nlocal: *;\n};\n")
endif()
