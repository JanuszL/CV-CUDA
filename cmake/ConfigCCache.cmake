# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

find_program(CCACHE_EXEC ccache)

if(CCACHE_EXEC)
    if(NOT CCACHE_STATSLOG)
        set(CCACHE_STATSLOG ${CMAKE_BINARY_DIR}/ccache_stats.log)
    endif()
    set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES ${CCACHE_STATSLOG})

    set(compiler_driver ${CMAKE_BINARY_DIR}/compiler_driver.sh)
    file(WRITE ${compiler_driver}
"#!/bin/bash
CCACHE_STATSLOG=${CCACHE_STATSLOG} ${CCACHE_EXEC} $@
")
    file(CHMOD ${compiler_driver} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)

    set(CMAKE_CXX_COMPILER_LAUNCHER ${compiler_driver})
    set(CMAKE_C_COMPILER_LAUNCHER ${compiler_driver})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${compiler_driver})
endif()
