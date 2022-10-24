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

#ifndef NVCV_UTIL_HASHMD5_HPP
#define NVCV_UTIL_HASHMD5_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nv::cv::util {

class HashMD5
{
public:
    HashMD5();
    HashMD5(const HashMD5 &) = delete;
    ~HashMD5();

    void                    operator()(const void *data, size_t lenBytes);
    std::array<uint8_t, 16> getHashAndReset();

    template<class T>
    void operator()(const T &value)
    {
        static_assert(std::has_unique_object_representations_v<T>, "Can't hash this type");
        this->operator()(&value, sizeof(value));
    }

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

template<class T>
std::enable_if_t<std::has_unique_object_representations_v<T>> Update(HashMD5 &hash, const T &value)
{
    hash(value);
}

template<class T>
void Update(HashMD5 &hash, const T *value)
{
    static_assert(sizeof(T) == 0, "Won't do md5 of a pointer");
}

void Update(HashMD5 &hash, const char *value);

template<std::ranges::range R>
void Update(nv::cv::util::HashMD5 &hash, const R &r)
{
    Update(hash, std::ranges::size(r));
    if constexpr (std::ranges::contiguous_range<
                      R> && std::has_unique_object_representations_v<std::ranges::range_value_t<R>>)
    {
        // It's faster to do this if range is contiguous and elements have unique object representation
        hash(std::ranges::data(r), std::ranges::size(r) * sizeof(std::ranges::range_value_t<R>));
    }
    else
    {
        // Must go one by one
        for (auto &v : r)
        {
            Update(hash, v);
        }
    }
}

template<class T1, class T2, class... TT>
void Update(HashMD5 &hash, const T1 &v1, const T2 &v2, const TT &...v)
{
    Update(hash, v1);
    Update(hash, v2);

    (..., Update(hash, v));
}

template<class T>
std::enable_if_t<std::is_floating_point_v<T>> Update(HashMD5 &hash, const T &value)
{
    hash(std::hash<T>()(value));
}

} // namespace nv::cv::util

namespace std {

template<typename... TT>
void Update(nv::cv::util::HashMD5 &hash, const tuple<TT...> &t)
{
    if constexpr (has_unique_object_representations_v<tuple<TT...>>)
    {
        return hash(t);
    }

    auto th = forward_as_tuple(hash);

    apply(nv::cv::util::Update<TT...>, tuple_cat(th, t));
};

inline void Update(nv::cv::util::HashMD5 &hash, const string &s)
{
    return hash(s.data(), s.size());
}

inline void Update(nv::cv::util::HashMD5 &hash, const string_view &s)
{
    return hash(s.data(), s.size());
}

inline void Update(nv::cv::util::HashMD5 &hash, const std::type_info &t)
{
    return hash(t.hash_code());
}

template<class T>
void Update(nv::cv::util::HashMD5 &hash, const optional<T> &o)
{
    using nv::cv::util::Update;

    // We can't rely on std::hash<T> for optionals because they
    // require a valid hash specialization for T. Since our
    // types use HashValue overloads, we have to do this instead.
    if (o)
    {
        return Update(hash, *o);
    }
    else
    {
        return Update(hash, std::hash<optional<int>>()(nullopt));
    }
}

} // namespace std

#endif // NVCV_UTIL_HASHMD5_HPP
