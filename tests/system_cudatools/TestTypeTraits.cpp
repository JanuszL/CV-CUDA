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

#include <common/TypedTests.hpp>    // for NVCV_TYPED_TEST_SUITE_F, etc.
#include <common/ValueTests.hpp>    // for StringLiteral
#include <nvcv/cuda/TypeTraits.hpp> // the object of this test

#include <limits> // for std::numeric_limits, etc.

namespace t    = ::testing;
namespace test = nv::cv::test;
namespace cuda = nv::cv::cuda;

template<int N>
using TStr = typename test::StringLiteral<N>;

// --------------------------- Testing TypeTraits ------------------------------

template<typename T>
class TypeTraitsBaseTest : public t::Test
{
public:
    using Type = T;
};

template<typename T>
class TypeTraitsSupportedBaseTest : public TypeTraitsBaseTest<T>
{
};

using TypeTraitsSupportedBaseTypes = t::Types<unsigned char, signed char, unsigned short, short, unsigned int, int,
                                              unsigned long, long, unsigned long long, long long, float, double>;

TYPED_TEST_SUITE(TypeTraitsSupportedBaseTest, TypeTraitsSupportedBaseTypes);

TYPED_TEST(TypeTraitsSupportedBaseTest, CorectTypeTraits)
{
    using TypeTraitsT = cuda::TypeTraits<typename TestFixture::Type>;

    EXPECT_TRUE((std::is_same<typename TestFixture::Type, typename TypeTraitsT::base_type>::value));

    EXPECT_FALSE(cuda::detail::IsCompound<typename TestFixture::Type>::value);

    EXPECT_EQ(TypeTraitsT::components, 0);

    EXPECT_EQ(TypeTraitsT::elements, 1);

    EXPECT_EQ(TypeTraitsT::min, std::numeric_limits<typename TestFixture::Type>::min());
    EXPECT_EQ(TypeTraitsT::max, std::numeric_limits<typename TestFixture::Type>::max());
}

// ---------------------------- Testing BaseType -------------------------------

template<typename T>
class TypeTraitsSupportedVectorTest : public TypeTraitsBaseTest<T>
{
};

using TypeTraitsSupportedVectorTypes
    = t::Types<uchar1, char2, ushort3, short4, uint1, int2, ulong3, long4, ulonglong1, longlong2, float3, double4>;

TYPED_TEST_SUITE(TypeTraitsSupportedVectorTest, TypeTraitsSupportedVectorTypes);

TYPED_TEST(TypeTraitsSupportedVectorTest, CorrectTypeTraits)
{
    using TypeTraitsT = cuda::TypeTraits<typename TestFixture::Type>;
    using BaseType    = cuda::BaseType<typename TestFixture::Type>;

    EXPECT_TRUE((std::is_same<BaseType, typename TypeTraitsT::base_type>::value));

    EXPECT_TRUE(cuda::detail::IsCompound<typename TestFixture::Type>::value);

    EXPECT_TRUE(TypeTraitsT::components >= 1);
    EXPECT_TRUE(TypeTraitsT::components <= 4);
    EXPECT_TRUE(TypeTraitsT::components == sizeof(typename TestFixture::Type) / sizeof(BaseType));

    EXPECT_TRUE(TypeTraitsT::elements >= 1);
    EXPECT_TRUE(TypeTraitsT::elements <= 4);
    EXPECT_TRUE(TypeTraitsT::elements == sizeof(typename TestFixture::Type) / sizeof(BaseType));

    EXPECT_EQ(TypeTraitsT::min, std::numeric_limits<typename TypeTraitsT::base_type>::min());
    EXPECT_EQ(TypeTraitsT::max, std::numeric_limits<typename TypeTraitsT::base_type>::max());
}

// ------------------- Testing TypeTraits with Type Qualifiers -----------------

class TypeTraitsTypeQualifierTest : public t::Test
{
};

TEST_F(TypeTraitsTypeQualifierTest, CorrectTypeTraitsAsIfNoQualifier)
{
    using Type = char;

    EXPECT_TRUE((std::is_same<cuda::BaseType<const Type>, cuda::BaseType<Type>>::value));
    EXPECT_TRUE((std::is_same<cuda::BaseType<volatile Type>, cuda::BaseType<Type>>::value));
    EXPECT_TRUE((std::is_same<cuda::BaseType<const volatile Type>, cuda::BaseType<Type>>::value));

    EXPECT_TRUE(cuda::TypeTraits<const Type>::components == cuda::TypeTraits<Type>::components);
    EXPECT_TRUE(cuda::TypeTraits<volatile Type>::components == cuda::TypeTraits<Type>::components);
    EXPECT_TRUE(cuda::TypeTraits<const volatile Type>::components == cuda::TypeTraits<Type>::components);

    EXPECT_TRUE(cuda::TypeTraits<const Type>::elements == cuda::TypeTraits<Type>::elements);
    EXPECT_TRUE(cuda::TypeTraits<volatile Type>::elements == cuda::TypeTraits<Type>::elements);
    EXPECT_TRUE(cuda::TypeTraits<const volatile Type>::elements == cuda::TypeTraits<Type>::elements);
}

// ---------------------------- Testing MakeType -------------------------------

template<class T>
class TypeTraitsMakeTypeVectorTest : public t::Test
{
public:
    using BaseType                     = test::type::GetType<T, 0>;
    static constexpr int NumComponents = test::type::GetValue<T, 1>;
    static constexpr int NumElements   = NumComponents == 0 ? 1 : NumComponents;
};

using MakeTypeSupportedVectorTypes = test::type::Combine<TypeTraitsSupportedBaseTypes, test::Values<0, 4>>;

NVCV_TYPED_TEST_SUITE_F(TypeTraitsMakeTypeVectorTest, MakeTypeSupportedVectorTypes);

TYPED_TEST(TypeTraitsMakeTypeVectorTest, CorrectTypeTraits)
{
    using CompoundType = cuda::MakeType<typename TestFixture::BaseType, TestFixture::NumComponents>;

    EXPECT_TRUE((std::is_same<typename TestFixture::BaseType, typename cuda::BaseType<CompoundType>>::value));

    EXPECT_TRUE(cuda::TypeTraits<CompoundType>::components == TestFixture::NumComponents);

    EXPECT_TRUE(cuda::TypeTraits<CompoundType>::elements == TestFixture::NumElements);

    EXPECT_EQ(TestFixture::NumComponents == 0,
              (cuda::detail::IsSame<typename TestFixture::BaseType, CompoundType>::value));
}

// -------------------- Testing MakeType with Type Qualifiers ------------------

template<typename T>
class TypeTraitsMakeTypeWithQualifierTest : public TypeTraitsMakeTypeVectorTest<T>
{
};

using MakeTypeSomeTypesWithQualifiers
    = test::type::Concat<test::type::Combine<test::Types<const int>, test::Values<0, 1>>,
                         test::type::Combine<test::Types<volatile int>, test::Values<0, 1>>,
                         test::type::Combine<test::Types<const volatile int>, test::Values<0, 1>>>;

NVCV_TYPED_TEST_SUITE_F(TypeTraitsMakeTypeWithQualifierTest, MakeTypeSomeTypesWithQualifiers);

TYPED_TEST(TypeTraitsMakeTypeWithQualifierTest, CorrectTypeQualifiers)
{
    using CompoundType = cuda::MakeType<typename TestFixture::BaseType, TestFixture::NumComponents>;

    EXPECT_EQ(std::is_const<typename TestFixture::BaseType>::value, std::is_const<CompoundType>::value);
    EXPECT_EQ(std::is_volatile<typename TestFixture::BaseType>::value, std::is_volatile<CompoundType>::value);
}

// ------------------------ Testing ConvertBaseTypeTo --------------------------

template<typename T>
class TypeTraitsConvertBaseTypeToTest : public TypeTraitsBaseTest<T>
{
};

NVCV_TYPED_TEST_SUITE_F(TypeTraitsConvertBaseTypeToTest, t::Types<char, short1, char2, uint3, double4>);

TYPED_TEST(TypeTraitsConvertBaseTypeToTest, CorrectTypeTraits)
{
    using FloatType = cuda::ConvertBaseTypeTo<float, typename TestFixture::Type>;

    EXPECT_TRUE((std::is_same<cuda::BaseType<FloatType>, float>::value));

    EXPECT_TRUE(cuda::TypeTraits<typename TestFixture::Type>::components == cuda::TypeTraits<FloatType>::components);

    EXPECT_TRUE(cuda::TypeTraits<typename TestFixture::Type>::elements == cuda::TypeTraits<FloatType>::elements);

    EXPECT_FALSE((cuda::detail::IsSame<FloatType, typename TestFixture::Type>::value));
}

// -------------- Testing ConvertBaseTypeTo with Type Qualifiers ---------------

template<typename T>
class TypeTraitsConvertBaseTypeToWithQualifiersTest : public TypeTraitsBaseTest<T>
{
};

NVCV_TYPED_TEST_SUITE_F(TypeTraitsConvertBaseTypeToWithQualifiersTest,
                        t::Types<const short3, volatile short3, const volatile short3>);

TYPED_TEST(TypeTraitsConvertBaseTypeToWithQualifiersTest, CorrectTypeQualifiers)
{
    using FloatType = cuda::ConvertBaseTypeTo<float, typename TestFixture::Type>;

    EXPECT_EQ(std::is_const<typename TestFixture::Type>::value, std::is_const<FloatType>::value);
    EXPECT_EQ(std::is_volatile<typename TestFixture::Type>::value, std::is_volatile<FloatType>::value);
}

// --------------------------- Testing GetElement ------------------------------

template<typename T>
class TypeTraitsGetElementTest : public TypeTraitsBaseTest<T>
{
public:
    using PixelType                  = typename TypeTraitsBaseTest<T>::Type;
    static constexpr int NumElements = cuda::TypeTraits<PixelType>::elements;

    PixelType pix;

    TypeTraitsGetElementTest()
    {
        if constexpr (NumElements == 4)
        {
            pix = {1, 2, 3, 4};
        }
        else if constexpr (NumElements == 3)
        {
            pix = {1, 2, 3};
        }
        else if constexpr (NumElements == 2)
        {
            pix = {1, 2};
        }
        else if constexpr (NumElements == 1)
        {
            pix = {1};
        }
    }
};

using SomeSupportedTypes = t::Types<char, ushort1, uchar2, int3, float4>;

TYPED_TEST_SUITE(TypeTraitsGetElementTest, SomeSupportedTypes);

TYPED_TEST(TypeTraitsGetElementTest, CorrectZeroInitializedChannels)
{
    for (int e = 0; e < this->NumElements; ++e)
    {
        EXPECT_EQ(nv::cv::cuda::GetElement(this->pix, e), e + 1);
    }
}

// ----------------------------- Testing SetAll --------------------------------

template<typename T>
class TypeTraitsSetAllTest : public TypeTraitsBaseTest<T>
{
};

NVCV_TYPED_TEST_SUITE_F(TypeTraitsSetAllTest, t::Types<char, short1, uchar2, int3, float4>);

TYPED_TEST(TypeTraitsSetAllTest, CorrectOutputOfSetAll)
{
    int  gold = 3;
    auto test = cuda::SetAll<typename TestFixture::Type>(gold);

    using TestType = decltype(test);

    EXPECT_TRUE((std::is_same<TestType, typename TestFixture::Type>::value));

    for (int c = 0; c < cuda::TypeTraits<TestType>::elements; ++c)
    {
        EXPECT_EQ(static_cast<int>(cuda::GetElement(test, c)), gold);
    }
}

// ------------------- Testing GetTypeName and operator << ---------------------

template<class T>
class TypeTraitsVectorTypePrintTest : public t::Test
{
public:
    using Type = test::type::GetType<T, 0>;

    static constexpr TStr GoldTypeName    = test::type::GetValue<T, 1>;
    static constexpr TStr GoldValueOutput = test::type::GetValue<T, 2>;

    Type val;

    TypeTraitsVectorTypePrintTest()
    {
        for (int e = 0; e < cuda::TypeTraits<Type>::elements; ++e)
        {
            cuda::GetElement(val, e) = e + 1;
        }
    }
};

NVCV_TYPED_TEST_SUITE_F(
    TypeTraitsVectorTypePrintTest,
    test::type::Zip<test::Types<float, double1, int2, short3, ulong4>,
                    test::Values<TStr("float"), TStr("double1"), TStr("int2"), TStr("short3"), TStr("ulong4")>,
                    test::Values<TStr("1"), TStr("double1(1)"), TStr("int2(1, 2)"), TStr("short3(1, 2, 3)"),
                                 TStr("ulong4(1, 2, 3, 4)")>>);

TYPED_TEST(TypeTraitsVectorTypePrintTest, CorrectPrintOfTypeAndValues)
{
    EXPECT_STREQ(nv::cv::cuda::GetTypeName<typename TestFixture::Type>(), this->GoldTypeName.value);

    std::ostringstream oss;

    EXPECT_NO_THROW(oss << this->val);

    EXPECT_STREQ(oss.str().c_str(), this->GoldValueOutput.value);
}
