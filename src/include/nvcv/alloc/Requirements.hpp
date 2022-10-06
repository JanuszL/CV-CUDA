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
 * @file Requirements.hpp
 *
 * @brief Defines the public C++ interface to NVCV resource requirements.
 *
 * Several objects in NVCV require resource allocation. Resource requirements
 * is a way for them to inform how many resources they need. This information
 * can be used by allocators to pre-allocate the resources that will be used.
 */

#ifndef NVCV_REQUIREMENTS_HPP
#define NVCV_REQUIREMENTS_HPP

#include "../detail/CheckError.hpp"
#include "Requirements.h"

namespace nv { namespace cv {

class Requirements final
{
public:
    class ConstMemory
    {
        friend class Requirements;
        ConstMemory(const NVCVMemRequirements &reqs);

    protected:
        const NVCVMemRequirements &m_reqs;

    public:
        static constexpr int size();
        int64_t              numBlocks(int log2BlockSizeBytes) const;

        const NVCVMemRequirements &cdata() const;
    };

    class Memory : public ConstMemory
    {
        friend class Requirements;
        Memory(NVCVMemRequirements &reqs);

    public:
        void addBuffer(int64_t bufSize, int64_t bufAlignment);

        using ConstMemory::cdata;
        NVCVMemRequirements &cdata();
    };

    Requirements();
    explicit Requirements(NVCVRequirements reqs);

    ConstMemory deviceMem() const;
    Memory      deviceMem();

    ConstMemory hostMem() const;
    Memory      hostMem();

    ConstMemory hostPinnedMem() const;
    Memory      hostPinnedMem();

    NVCVRequirements       &cdata();
    const NVCVRequirements &cdata() const;

    Requirements &operator+=(const Requirements &that);
    Requirements  operator+(const Requirements &that) const;

private:
    NVCVRequirements m_reqs;
};

int64_t CalcTotalSizeBytes(const Requirements::ConstMemory &mem);

// Implementation

inline Requirements::ConstMemory::ConstMemory(const NVCVMemRequirements &reqs)
    : m_reqs(reqs)
{
}

inline constexpr int Requirements::ConstMemory::size()
{
    return NVCV_MAX_MEM_REQUIREMENTS_LOG2_BLOCK_SIZE;
}

inline const NVCVMemRequirements &Requirements::ConstMemory::cdata() const
{
    return m_reqs;
}

inline int64_t CalcTotalSizeBytes(const Requirements::ConstMemory &mem)
{
    int64_t sizeBytes;
    detail::CheckThrow(nvcvMemRequirementsCalcTotalSizeBytes(&mem.cdata(), &sizeBytes));
    return sizeBytes;
}

inline int64_t Requirements::ConstMemory::numBlocks(int log2BlockSizeBytes) const
{
    assert(0 <= log2BlockSizeBytes && log2BlockSizeBytes < this->size());
    return this->cdata().numBlocks[log2BlockSizeBytes];
}

inline Requirements::Memory::Memory(NVCVMemRequirements &reqs)
    : ConstMemory(reqs)
{
}

inline NVCVMemRequirements &Requirements::Memory::cdata()
{
    return const_cast<NVCVMemRequirements &>(m_reqs);
}

inline void Requirements::Memory::addBuffer(int64_t bufSize, int64_t bufAlign)
{
    detail::CheckThrow(nvcvMemRequirementsAddBuffer(&this->cdata(), bufSize, bufAlign));
}

inline Requirements::Requirements()
{
    detail::CheckThrow(nvcvRequirementsInit(&m_reqs));
}

inline Requirements::Requirements(NVCVRequirements reqs)
    : m_reqs(std::move(reqs))
{
}

inline auto Requirements::deviceMem() const -> ConstMemory
{
    return ConstMemory{m_reqs.deviceMem};
}

inline auto Requirements::deviceMem() -> Memory
{
    return Memory{m_reqs.deviceMem};
}

inline auto Requirements::hostMem() const -> ConstMemory
{
    return ConstMemory{m_reqs.hostMem};
}

inline auto Requirements::hostMem() -> Memory
{
    return Memory{m_reqs.hostMem};
}

inline auto Requirements::hostPinnedMem() const -> ConstMemory
{
    return ConstMemory{m_reqs.hostPinnedMem};
}

inline auto Requirements::hostPinnedMem() -> Memory
{
    return Memory{m_reqs.hostPinnedMem};
}

inline NVCVRequirements &Requirements::cdata()
{
    return m_reqs;
}

inline const NVCVRequirements &Requirements::cdata() const
{
    return m_reqs;
}

inline Requirements &Requirements::operator+=(const Requirements &that)
{
    detail::CheckThrow(nvcvRequirementsAdd(&m_reqs, &that.cdata()));
    return *this;
}

inline Requirements Requirements::operator+(const Requirements &that) const
{
    return Requirements{*this} += that;
}

}} // namespace nv::cv

#endif // NVCV_REQUIREMENTS_HPP
