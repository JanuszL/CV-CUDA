# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# We must run the following at "include" time, not at function call time,
# to find the path to this module rather than the path to a calling list file
get_filename_component(config_version_script_path ${CMAKE_CURRENT_LIST_FILE} PATH)

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC REPO_COMMIT)

set(PROJECT_VERSION "${PROJECT_VERSION}${PROJECT_VERSION_SUFFIX}")

function(configure_version target libprefix incpath VERSION_FULL)
    string(TOUPPER "${target}" TARGET)
    string(TOUPPER "${libprefix}" LIBPREFIX)

    string(REGEX MATCH "-(.*)$" version_suffix "${VERSION_FULL}")
    set(VERSION_SUFFIX ${CMAKE_MATCH_1})

    string(REGEX MATCHALL "[0-9]+" version_list "${VERSION_FULL}")
    list(GET version_list 0 VERSION_MAJOR)
    list(GET version_list 1 VERSION_MINOR)
    list(GET version_list 2 VERSION_PATCH)

    list(LENGTH version_list num_version_components)

    if(num_version_components EQUAL 3)
        set(VERSION_TWEAK 0)
    elseif(num_version_components EQUAL 4)
        list(GET version_list 3 VERSION_TWEAK)
    else()
        message(FATAL_ERROR "Version must have either 3 or 4 components")
    endif()

    math(EXPR VERSION_API_CODE "${VERSION_MAJOR}*100 + ${VERSION_MINOR}")

    string(REPLACE "-" "_" tmp ${VERSION_FULL})
    set(VERSION_BUILD "${tmp}-${CVCUDA_BUILD_SUFFIX}")

    set(LIBPREFIX ${LIBPREFIX})

    configure_file(${config_version_script_path}/VersionDef.h.in include/${incpath}/VersionDef.h @ONLY ESCAPE_QUOTES)
    configure_file(${config_version_script_path}/VersionUtils.h.in include/${incpath}/detail/VersionUtils.h @ONLY ESCAPE_QUOTES)

    set(${LIBPREFIX}_VERSION_FULL ${VERSION_FULL} CACHE INTERNAL "${TARGET} full version")
    set(${LIBPREFIX}_VERSION_MAJOR ${VERSION_MAJOR} CACHE INTERNAL "${TARGET} major version")
    set(${LIBPREFIX}_VERSION_MINOR ${VERSION_MINOR} CACHE INTERNAL "${TARGET} minor version")
    set(${LIBPREFIX}_VERSION_PATCH ${VERSION_PATCH} CACHE INTERNAL "${TARGET} patch version")
    set(${LIBPREFIX}_VERSION_TWEAK ${VERSION_TWEAK} CACHE INTERNAL "${TARGET} tweak version")
    set(${LIBPREFIX}_VERSION_SUFFIX ${VERSION_SUFFIX} CACHE INTERNAL "${TARGET} version suffix")
    set(${LIBPREFIX}_VERSION_API ${VERSION_MAJOR}.${VERSION_MINOR} CACHE INTERNAL "${TARGET} API version")
    set(${LIBPREFIX}_VERSION_API_CODE ${VERSION_API_CODE} CACHE INTERNAL "${TARGET} API code")
    set(${LIBPREFIX}_VERSION_BUILD ${VERSION_BUILD} CACHE INTERNAL "${TARGET} build version")

    # So that the generated headers are found
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/VersionDef.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${incpath}
            COMPONENT dev)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/detail/VersionUtils.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${incpath}/detail
            COMPONENT dev)
endfunction()

function(configure_symbol_versioning target libprefix incpath version)
    string(TOUPPER "${target}" TARGET)
    string(TOUPPER "${libprefix}" LIBPREFIX)

    string(REGEX MATCHALL "[0-9]+" version_list "${version}")
    list(GET version_list 0 VERSION_MAJOR)
    list(GET version_list 1 VERSION_MINOR)
    list(GET version_list 2 VERSION_PATCH)

    set_target_properties(${target} PROPERTIES
        VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
        SOVERSION "${VERSION_MAJOR}"
    )

    # Create exports file for symbol versioning ---------------------------------
    set(EXPORTS_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/exports.ldscript")
    target_link_libraries(${target}
        PRIVATE
        -Wl,--version-script ${EXPORTS_OUTPUT}
    )
    get_target_property(TARGET_SOURCES ${target} SOURCES)
    set(GEN_EXPORTS_SCRIPT "${config_version_script_path}/CreateExportsFile.cmake")

    add_custom_command(OUTPUT ${EXPORTS_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
                                 -DSOURCES="${TARGET_SOURCES}"
                                 -DOUTPUT=${EXPORTS_OUTPUT}
                                 -P "${GEN_EXPORTS_SCRIPT}"
        DEPENDS ${GEN_EXPORTS_SCRIPT} ${TARGET_SOURCES})

    add_custom_target(create_${target}_exports_file DEPENDS ${EXPORTS_OUTPUT})
    add_dependencies(${target} create_${target}_exports_file)

    #   Configure symbol visibility ---------------------------------------------
    set_target_properties(${target} PROPERTIES VISIBILITY_INLINES_HIDDEN on
                                               C_VISIBILITY_PRESET hidden
                                               CXX_VISIBILITY_PRESET hidden
                                               CUDA_VISIBILITY_PRESET hidden)

    configure_file(${config_version_script_path}/Export.h.in include/${incpath}/detail/Export.h @ONLY ESCAPE_QUOTES)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/${incpath}/detail/Export.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${incpath}/detail
            COMPONENT dev)

    target_compile_definitions(${target} PRIVATE -D${LIBPREFIX}_EXPORTING=1)
endfunction()
