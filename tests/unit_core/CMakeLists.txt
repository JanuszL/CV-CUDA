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

add_executable(nvcv_test_unit_core
    ../Main.cpp

    TestHashMD5.cpp
    TestValueList.cpp
    TestValueTests.cpp
    TestTypeList.cpp
    TestTypedTests.cpp
    TestCheckError.cpp
    TestString.cpp
    TestStaticVector.cpp
    TestVersion.cpp
    TestIndexSequence.cpp
    TestMath.cpp
    TestMetaprogramming.cpp
    TestOptional.cpp
    TestLegacyHelpers.cpp
    TestLockFreeStack.cpp
    TestHandleManager.cpp
    TestTensorDataUtils.cpp
    TestAlgorithm.cpp
    TestCompat.cpp
)

target_compile_definitions(nvcv_test_unit_core
    PRIVATE
        -DNVCV_UNIT_TEST=1
)

target_link_libraries(nvcv_test_unit_core
    PRIVATE
        nvcv_util
        nvcv_test_common
        nvcv_private_core
        nvcv_private_operators
)

nvcv_add_test(nvcv_test_unit_core)
