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

#include "HashMD5.hpp"

#include "Assert.h"

#include <openssl/evp.h>

#include <cstring>

namespace nv::cv::util {

struct HashMD5::Impl
{
    EVP_MD_CTX *ctx;
};

HashMD5::HashMD5()
    : pimpl{std::make_unique<Impl>()}
{
    pimpl->ctx = EVP_MD_CTX_create();
    NVCV_ASSERT(pimpl->ctx != nullptr);

    int ret = EVP_DigestInit_ex(pimpl->ctx, EVP_md5(), NULL);
    NVCV_ASSERT(ret == 1);
}

HashMD5::~HashMD5()
{
    EVP_MD_CTX_destroy(pimpl->ctx);
}

void HashMD5::operator()(const void *data, size_t lenBytes)
{
    int ret = EVP_DigestUpdate(pimpl->ctx, data, lenBytes);
    NVCV_ASSERT(ret == 1);
}

std::array<uint8_t, 16> HashMD5::getHashAndReset()
{
    unsigned char buf[EVP_MAX_MD_SIZE];
    unsigned int  nwritten = sizeof(buf);
    // it also resets the context
    int           ret = EVP_DigestFinal(pimpl->ctx, buf, &nwritten);
    NVCV_ASSERT(ret == 1);

    // Be ready for a new run
    ret = EVP_DigestInit_ex(pimpl->ctx, EVP_md5(), NULL);
    NVCV_ASSERT(ret == 1);

    NVCV_ASSERT(nwritten == 16);
    std::array<uint8_t, 16> hash;
    memcpy(&hash[0], buf, sizeof(hash));
    return hash;
}

void Update(HashMD5 &hash, const char *value)
{
    if (value)
    {
        Update(hash, std::string_view(value));
    }
    else
    {
        Update(hash, -28374);
    }
}

} // namespace nv::cv::util
