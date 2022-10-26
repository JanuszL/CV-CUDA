/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NVCV_TEST_COMMON_UTILS_HPP
#define NVCV_TEST_COMMON_UTILS_HPP

#include <nvcv/Tensor.hpp> // for ITensorDataPitchDevice, etc.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace nv::cv::test {

#ifdef DEBUG_PRINT_IMAGE
template<typename T>
void DebugPrintImage(std::vector<T> &in, int rowPitch)
{
    std::cout << "\nDebug print image: row pitch = " << rowPitch << " total size = " << in.size() << "\n";
    for (size_t i = 0; i < in.size(); i++)
    {
        if (i % rowPitch == 0)
        {
            std::cout << "\n";
        }
        std::cout << " " << std::fixed << std::setprecision(2) << std::setw(3)
                  << static_cast<std::conditional_t<(sizeof(T) == 1), int, T>>(in[i]);
    }
    std::cout << "\n";
}
#endif
#ifdef DEBUG_PRINT_DIFF
template<typename T>
static void DebugPrintDiff(std::vector<T> &test, std::vector<T> &gold)
{
    using TT = std::conditional_t<(sizeof(T) == 1), int, T>;
    std::cout << "\nDebug print diff:\n";
    for (size_t i = 0; i < test.size(); i++)
    {
        if (test[i] != gold[i])
        {
            std::cout << "[at " << i << ": test = " << static_cast<TT>(test[i])
                      << " x gold = " << static_cast<TT>(gold[i]) << "]\n";
        }
    }
    std::cout << "\n";
}
#endif

template<typename T>
void FillRandomData(std::vector<T> &hDst, T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max())
{
    using distribution = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>,
                                            std::uniform_real_distribution<T>>;

    static std::default_random_engine reng{0};

    distribution dist{min, max};

    std::generate(hDst.begin(), hDst.end(), [&] { return dist(reng); });
}

} // namespace nv::cv::test

#endif // NVCV_TEST_COMMON_TYPELIST_HPP
