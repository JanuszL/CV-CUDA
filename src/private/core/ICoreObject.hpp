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

#ifndef NVCV_PRIV_ICOREOBJECT_HPP
#define NVCV_PRIV_ICOREOBJECT_HPP

#include "Exception.hpp"
#include "Version.hpp"

#include <type_traits>

// Here we define the base classes for all objects that can be created/destroyed
// by the user, the so-called "core objects".
//
// IMPORTANT: their dynamic definitions must NEVER change, as we must keep backward ABI compatibility
// in case users are mixing objects created by different NVCV versions. This is
// valid even between different major versions.

// Interface classes that inherit from ICoreObject must also obey some rules:
// 1. Once the interface is released to public, its dynamic definition
//    must never change if major versions are kept the same. To change them, major
//    NVCV version must be bumped.
// 2. If new virtual methods need to be added to an interface, a new interface that
//    inherits from the previous one must be created. Code that need the new interface
//    must do down casts.

namespace nv::cv::priv {

class ICoreObject
{
public:
    // Disable copy/move to avoid slicing.
    ICoreObject(const ICoreObject &) = delete;

    virtual ~ICoreObject() = default;

    Version version() const
    {
        return doGetVersion();
    }

protected:
    ICoreObject() = default;

    // Using NVI idiom.
    virtual Version doGetVersion() const = 0;
};

// Base class for all core objects that exposes a handle with a particular type.
// Along with the ToPtr and ToRef methods below, it provides facilities to convert
// between external C handles to the actual internal object instance they refer to.
template<class I, class HANDLE>
class ICoreObjectHandle : public ICoreObject
{
public:
    using HandleType    = HANDLE;
    using InterfaceType = I;

    HandleType handle() const
    {
        return reinterpret_cast<HandleType>(const_cast<ICoreObject *>(static_cast<const ICoreObject *>(this)));
    }
};

template<class T>
T *ToPtr(typename T::HandleType h)
{
    using ICore = std::conditional_t<std::is_const_v<T>, const ICoreObject, ICoreObject>;

    // First cast to the core interface, this must always succeed.
    if (ICore *core = reinterpret_cast<ICore *>(h))
    {
        // If major version are the same,
        if (core->version().major() == CURRENT_VERSION.major())
        {
            // We now can cat to the required type.
            return dynamic_cast<T *>(core);
        }
        else
        {
            throw Exception(NVCV_ERROR_NOT_COMPATIBLE)
                << "Object version " << core->version() << " not compatible with NVCV version " << CURRENT_VERSION;
        }
    }
    else
    {
        return nullptr;
    }
}

template<class T>
T &ToRef(typename T::HandleType h)
{
    if (h == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Handle cannot be NULL");
    }

    if (T *child = ToPtr<T>(h))
    {
        return *child;
    }
    else
    {
        throw Exception(NVCV_ERROR_NOT_COMPATIBLE,
                        "Handle doesn't correspond to the requested object, or versions are incompatible.");
    }
}

} // namespace nv::cv::priv

#endif // NVCV_PRIV_ICOREOBJECT_HPP
