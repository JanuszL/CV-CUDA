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


# Create the binary packages for all python versions supported
foreach(VER ${PYTHON_VERSIONS})
    string(REPLACE "." "" VERNAME ${VER})

    # Let's get the python module file name for current python version
    if(USE_DEFAULT_PYTHON)
        set(PYTHON${VERNAME}_EXEC ${Python_EXECUTABLE})
    else()
        find_program(PYTHON${VERNAME}_EXEC python${VER} REQUIRED)
    endif()
    set(PYTHON_EXEC ${PYTHON${VERNAME}_EXEC})
    message("${VER}: ${PYTHON_EXEC}")
    execute_process(COMMAND ${PYTHON_EXEC} -c "from distutils import sysconfig as s;print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
                    OUTPUT_VARIABLE module_suffix
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    COMMAND_ERROR_IS_FATAL ANY)

    set(module_filename nvcv${module_suffix})

    set(python_module_name python${VER})
    # NSIS doesn't like the dot
    string(REPLACE "." "" python_module_name "${python_module_name}")
    string(TOUPPER ${python_module_name} PYTHON_MODULE_NAME)

    # Issue the install command for the python module
    install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${module_filename}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
            COMPONENT ${python_module_name})

    # Sets python module's RPATH to point to where nvcv library is.
    set(modpath
        "\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/python/${module_filename}")
    install(CODE "execute_process(COMMAND patchelf --set-rpath \"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\" \"${modpath}\" ERROR_VARIABLE error_patch)\nif(error_path)\n    message(FATAL_ERROR \"Error setting rpath of ${modpath}: ${error_path}\")\nendif()"
        COMPONENT ${python_module_name})

    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_DISABLED true)
    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_DISPLAY_NAME "Python ${VER}")
    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_DESCRIPTION "NVIDIA NVCV python ${VER} bindings")
    set(CPACK_COMPONENT_${PYTHON_MODULE_NAME}_GROUP python)

    if(UNIX)
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_NAME python${VER}-${CPACK_PACKAGE_NAME})

        set(NVCV_${PYTHON_MODULE_NAME}_FILE_NAME "nvcv-python${VER}-${PACKAGE_FULL_VERSION}")
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_FILE_NAME "${NVCV_${PYTHON_MODULE_NAME}_FILE_NAME}.deb")
        set(CPACK_ARCHIVE_${PYTHON_MODULE_NAME}_FILE_NAME "${NVCV_${PYTHON_MODULE_NAME}_FILE_NAME}")

        # Depend on current or any future ABI with same major version
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_DEPENDS "lib${nvcv_PACKAGE_NAME} (>= ${cvcuda_API_VERSION})")

        # Depend on python interpreter
        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_DEPENDS "${CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_DEPENDS}, python${VER}")

        # Set up debian control files
        set(SRC_PYTHON_MODULE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/python/${module_filename})
        set(DST_PYTHON_MODULE /usr/lib/python3/dist-packages/${module_filename})

        configure_file(${cvcuda_SOURCE_DIR}/cpack/debian_python_postinst.in ${cvcuda_BINARY_DIR}/cpack/python${VER}/postinst @ONLY)
        configure_file(${cvcuda_SOURCE_DIR}/cpack/debian_python_prerm.in ${cvcuda_BINARY_DIR}/cpack/python${VER}/prerm @ONLY)

        set(CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_CONTROL_EXTRA
            "${cvcuda_BINARY_DIR}/cpack/python${VER}/postinst"
            "${cvcuda_BINARY_DIR}/cpack/python${VER}/prerm")
    endif()

    if(BUILD_TESTS)
        set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS
                "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS},
                 ${CPACK_DEBIAN_${PYTHON_MODULE_NAME}_PACKAGE_NAME} (>= ${cvcuda_API_VERSION})")

        # For some reason these are needed with python-3.7
        if(VER VERSION_EQUAL "3.7")
            set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS
                    "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS}
                     , python3-typing-extensions")
        endif()
    endif()

    list(APPEND CPACK_COMPONENTS_ALL ${python_module_name})
endforeach()

if(BUILD_TESTS)
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS
            "${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS}, python3-pytest")
endif()
