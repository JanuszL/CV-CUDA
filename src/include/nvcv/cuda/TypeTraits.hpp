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

/**
 * @file TypeTraits.hpp
 *
 * @brief Defines type traits to handle regular C and CUDA compound types.
 */

#ifndef NVCV_CUDA_TYPE_TRAITS_HPP
#define NVCV_CUDA_TYPE_TRAITS_HPP

#include "detail/Metaprogramming.hpp" // for detail::TypeTraits, etc.

#include <cassert> // for assert, etc.
#include <ostream> // for std::ostream, etc.

namespace nv::cv::cuda {

/**
 * @brief Metastruct to define type traits for regular C types and CUDA built-in vector types
 *
 * @details CUDA built-in vector types are also called compound types.  The number of components in the metastruct
 * is zero for regular C types and between 1 and 4 for CUDA built-in vector types.  On the flip side, the number of
 * elements is between 1 and 4 for regular C types and CUDA built-in vector types.  The number of components may be
 * used to check if a type is of regular type (is zero) or compound type (is greater than zero).  The number of
 * elements may be used regardless of the type.  The base type of a CUDA built-in vector type is the type of each
 * of its components, for instance uchar4 has 4 elements of base type unsigned char.
 *
 * @code
 * int nc = nv::cv::cuda::TypeTraits<T>::components;
 * int ne = nv::cv::cuda::TypeTraits<T>::elements;
 * @endcode
 *
 * @tparam T Type to get traits from
 */
template<class T>
using TypeTraits = detail::TypeTraits<T>;

/**
 * @brief Metatype to get the base type of a CUDA built-in vector types
 *
 * @code
 * using PixelType = ...;
 * using ChannelType = nv::cv::cuda::BaseType<PixelType>;
 * @endcode
 *
 * @note This is identity for regular C types.
 *
 * @tparam T Type to get the base type from
 */
template<class T>
using BaseType = typename TypeTraits<T>::base_type;

/**
 * @brief Metatype to make a type from a base type and number of components
 *
 * @details When number of components is zero, it yields the identity (regular C) type, and when it is between 1
 * and 4 it yields the CUDA built-in compound (or vector) type.
 *
 * @code
 * using RGB8Type = MakeType<unsigned char, 3>; // yields uchar3
 * @endcode
 *
 * @note Note that T=char might yield uchar1..4 types when char is equal unsigned char, i.e. CHAR_MIN == 0.
 *
 * @tparam T Base type to make the type from
 * @tparam C Number of components to make the type
 */
template<class T, int C>
using MakeType = typename detail::MakeType<T, C>::type;

/**
 * @brief Metatype to convert the base type of a type
 *
 * @details The base type of target type \p T is replaced to be \p BT.
 *
 * @code
 * using PixelType = ...;
 * using FloatPixelType = ConvertBaseTypeTo<float, PixelType>; // yields float1..4
 * @endcode
 *
 * @tparam BT Base type to use in the conversion
 * @tparam T Target type to convert its base type
 */
template<class BT, class T>
using ConvertBaseTypeTo = typename detail::ConvertBaseTypeTo<BT, T>::type;

/**
 * @brief Metafunction to get an element by reference from a given value reference
 *
 * @details The value may be of CUDA compound type with 1 to 4 elements, where the corresponding element index is 0
 * to 3, and the return is a reference to the element with the base type of the compound type, copying the
 * constness (that is the return reference is constant if the input value is constant).  The value may be a regular
 * C type, in which case the element index is ignored and the identity is returned.
 *
 * @code
 * using PixelRGB8Type = MakeType<unsigned char, 3>;
 * PixelRGB8Type pix = ...;
 * auto green = GetElement(pix, 1); // yields unsigned char
 * @endcode
 *
 * @tparam T Type of the value to get the element from
 *
 * @param[in] v Value of type T to get an element from
 * @param[in] eidx Element index in [0, 3] inside the compound value to get the reference from
 *                 This element index is ignored in case the value is not of a CUDA compound type
 *
 * @return The reference of the value's element
 */
template<typename T, typename RT = detail::CopyConstness_t<T, BaseType<T>>>
__host__ __device__ typename std::enable_if<detail::IsCompound<T>::value, RT &>::type GetElement(T &v, int eidx)
{
    assert(eidx < TypeTraits<T>::elements);
    return reinterpret_cast<RT *>(&v)[eidx];
}

template<typename T, typename RT = detail::CopyConstness_t<T, T>>
__host__ __device__ typename std::enable_if<!detail::IsCompound<T>::value, RT &>::type GetElement(T &v, int)
{
    return v;
}

/**
 * @brief Metafunction to set all elements to the same value
 *
 * @details Set all elements to the value \p x passed as argument.  For instance, an int3 can have all its elements
 * set to zero by calling SetAll and passing int3 as template argument and zero as argument (see example below).
 *
 * @code
 * auto idx = SetAll<int3>(0); // starts an index int3 local variable with all elements zeros, as {0, 0, 0}
 * @endcode
 *
 * @tparam T Type to be returned with all elements set to the given value \p x
 *
 * @param[in] x Value to set all elements to
 *
 * @return The object of type T with all elements set to \p x
 */
template<typename T, class Req = detail::RequireHasTypeTraits<T>>
__host__ __device__ T SetAll(BaseType<T> x)
{
    T out{};

#pragma unroll
    for (int c = 0; c < TypeTraits<T>::elements; ++c)
    {
        GetElement(out, c) = x;
    }

    return out;
}

/**
 * @brief Metafunction to get the name of a type
 *
 * @details Unfortunately typeid().name() in C/C++ typeinfo yields different names depending on the platform.  This
 * function returns the name of the type resembling the CUDA compound type, that may be useful for debug printing.
 *
 * @code
 * std::cout << GetTypeName<PixelType>();
 * @endcode
 *
 * @tparam T Type to get the name from
 *
 * @return String with the name of the type
 */
template<class T, class Req = detail::RequireHasTypeTraits<T>>
__host__ __device__ const char *GetTypeName()
{
    return TypeTraits<T>::name;
}

} // namespace nv::cv::cuda

/**
 * @brief Metaoperator to insert a pixel into an output stream
 *
 * @details The pixel may be a CUDA compound type with 1 to 4 components.  This operator returns the output stream
 * changed by an additional string with the name of the type followed by each component value in between
 * parentheses.
 *
 * @code
 * PixelType pix = ...;
 * std::cout << pix;
 * @endcode
 *
 * @tparam T Type of the pixel to be inserted in the output stream
 *
 * @param[in, out] out Output stream to be changed and returned
 * @param[in] v Pixel value to be inserted formatted in the output stream
 *
 * @return Output stream with the pixel type and values
 */
template<class T, class Req = nv::cv::cuda::detail::RequireIsCompound<T>>
__host__ std::ostream &operator<<(std::ostream &out, const T &v)
{
    using BT         = nv::cv::cuda::BaseType<T>;
    using OutType    = typename std::conditional<sizeof(BT) == 1, int, BT>::type;
    constexpr int NC = nv::cv::cuda::TypeTraits<T>::components;

    out << nv::cv::cuda::GetTypeName<T>() << "(";

    for (int c = 0; c < NC; ++c)
    {
        if (c > 0)
        {
            out << ", ";
        }
        out << static_cast<OutType>(nv::cv::cuda::GetElement(v, c));
    }

    out << ")";

    return out;
}

#endif // NVCV_CUDA_TYPE_TRAITS_HPP
