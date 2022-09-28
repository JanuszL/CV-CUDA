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

document.addEventListener("DOMContentLoaded", function() {
    var params = window.location.search.substring(1).split("&").reduce(function(params, param) {
        if (!param)
        {
            return params;
        }

        var values = param.split("=");
        var name = values[0];
        var value = values[1];
        params[name] = value;
        return params;
    }, {});

    var form = document.getElementById("feedback-form");
    for (var name in params)
    {
        var input = form.querySelector("[name=" + name + "]");
        input.value = params[name];
    }
});
