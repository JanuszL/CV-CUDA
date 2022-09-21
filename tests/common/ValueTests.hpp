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

#ifndef NVCV_TEST_COMMON_VALUETESTS_HPP
#define NVCV_TEST_COMMON_VALUETESTS_HPP

#include "ValueList.hpp"

namespace nv::cv::test {

namespace detail {

template<class P>
std::string GetTestParamHashHelper(const P &info)
{
    // Let's use a hash of the parameter set as index.
    util::HashMD5 hash;
    Update(hash, info);

    // We don't need 64 bit worth of variation, 32-bit is enough and leads
    // to shorter suffixes.

    union Cast
    {
        uint8_t  array[16];
        uint64_t value[2];
    };

    static_assert(sizeof(hash.getHashAndReset()) == sizeof(Cast::array));

    Cast caster;
    memcpy(caster.array, &hash.getHashAndReset()[0], sizeof(caster.array));

    uint64_t code64 = caster.value[0] ^ caster.value[1];
    uint32_t code32 = (code64 & UINT32_MAX) ^ (code64 >> 32);

    std::ostringstream out;
    out << std::hex << std::setw(sizeof(code32) * 2) << std::setfill('0') << code32;
    return out.str();
}

} // namespace detail

template<class P>
std::string GetTestParamHash(const ::testing::WithParamInterface<P> &info)
{
    return detail::GetTestParamHashHelper(info.GetParam());
}

// We don't use googletest's default test suffix generator (that ascending number)
// because they aren't tied with the test parameter. If some platform doesn't have
// a particular test parameter (not supported?), the number will refer to a
// different parameter. We need the whole test name to be associated with the
// same test instance no matter what.
struct TestSuffixPrinter
{
    template<class P>
    std::string operator()(const ::testing::TestParamInfo<P> &info) const
    {
        return detail::GetTestParamHashHelper(info.param);
    }
};

} // namespace nv::cv::test

#define NVCV_INSTANTIATE_TEST_SUITE_P(GROUP, TEST, ...)                                                           \
    INSTANTIATE_TEST_SUITE_P(                                                                                     \
        GROUP, TEST,                                                                                              \
        ::testing::ValuesIn(typename ::nv::cv::test::detail::NormalizeValueList<                                  \
                            ::nv::cv::test::ValueList<typename TEST::ParamType>>::type(UniqueSort(__VA_ARGS__))), \
        ::nv::cv::test::TestSuffixPrinter())

#define NVCV_TEST_SUITE_P(TEST, ...)                                                              \
    static ::nv::cv::test::ValueList g_##TEST##_Params = ::nv::cv::test::UniqueSort(__VA_ARGS__); \
    class TEST : public ::testing::TestWithParam<decltype(g_##TEST##_Params)::value_type>         \
    {                                                                                             \
    protected:                                                                                    \
        template<int I>                                                                           \
        auto GetParamValue() const                                                                \
        {                                                                                         \
            return std::get<I>(GetParam());                                                       \
        }                                                                                         \
    };                                                                                            \
    NVCV_INSTANTIATE_TEST_SUITE_P(_, TEST, g_##TEST##_Params)

#endif // NVCV_TEST_COMMON_VALUETESTS_HPP
