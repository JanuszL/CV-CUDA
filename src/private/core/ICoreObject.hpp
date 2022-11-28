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
#include "HandleManager.hpp"
#include "Version.hpp"

#include <nvcv/alloc/Fwd.h>

#include <memory>
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

class alignas(kResourceAlignment) ICoreObject
{
public:
    // Disable copy/move to avoid slicing.
    ICoreObject(const ICoreObject &) = delete;

    virtual ~ICoreObject() = default;

    virtual Version version() const = 0;

protected:
    ICoreObject() = default;
};

template<class HANDLE>
class IHandleHolder
{
public:
    using HandleType = HANDLE;

    virtual void       setHandle(HandleType h) = 0;
    virtual HandleType handle()                = 0;
};

// Base class for all core objects that exposes a handle with a particular type.
// Along with the ToPtr and ToRef methods below, it provides facilities to convert
// between external C handles to the actual internal object instance they refer to.
template<class I, class HANDLE>
class ICoreObjectHandle
    : public ICoreObject
    , public IHandleHolder<HANDLE>
{
public:
    using InterfaceType = I;
};

template<class Interface>
class CoreObjectBase : public Interface
{
public:
    using HandleType = typename Interface::HandleType;

    void setHandle(HandleType h) final
    {
        m_handle = h;
    }

    HandleType handle() final
    {
        return m_handle;
    }

    cv::priv::Version version() const final
    {
        //todo need to have a version decoupled from NVCV
        return cv::priv::CURRENT_VERSION;
    }

private:
    HandleType m_handle = {};
};

template<class HandleType>
class CoreObjManager;

template<class T, class... ARGS>
typename T::HandleType CreateCoreObject(ARGS &&...args)
{
    using Manager = CoreObjManager<typename T::HandleType>;

    auto &mgr = Manager::Instance();

    typename T::HandleType h = mgr.template create<T>(std::forward<ARGS>(args)...);
    mgr.validate(h)->setHandle(h);
    return h;
}

template<class HandleType>
void DestroyCoreObject(HandleType handle)
{
    using Manager = CoreObjManager<HandleType>;

    auto &mgr = Manager::Instance();

    mgr.destroy(handle);
}

template<class, class = void>
constexpr bool HasObjManager = false;

template<class T>
constexpr bool HasObjManager<T, std::void_t<decltype(sizeof(CoreObjManager<T>))>> = true;

template<class HandleType>
inline ICoreObject *ToCoreObjectPtr(HandleType h)
{
    ICoreObject *core;

    if constexpr (HasObjManager<HandleType>)
    {
        using Manager = CoreObjManager<HandleType>;

        core = Manager::Instance().validate(h);
    }
    else
    {
        // First cast to the core interface, this must always succeed.
        core = reinterpret_cast<ICoreObject *>(h);
    }

    if (core != nullptr)
    {
        // If major version are the same,
        if (core->version().major() == CURRENT_VERSION.major())
        {
            return core;
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
T *ToStaticPtr(typename T::HandleType h)
{
    return static_cast<T *>(ToCoreObjectPtr(h));
}

template<class T>
T &ToStaticRef(typename T::HandleType h)
{
    T *child = ToStaticPtr<T>(h);

    if (child == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Handle was already destroyed");
    }

    return *child;
}

template<class T>
T *ToDynamicPtr(typename T::HandleType h)
{
    return dynamic_cast<T *>(ToCoreObjectPtr(h));
}

template<class T>
T &ToDynamicRef(typename T::HandleType h)
{
    if (T *child = ToDynamicPtr<T>(h))
    {
        return *child;
    }
    else
    {
        throw Exception(NVCV_ERROR_NOT_COMPATIBLE,
                        "Handle doesn't correspond to the requested object or was already destroyed.");
    }
}

} // namespace nv::cv::priv

#endif // NVCV_PRIV_ICOREOBJECT_HPP
