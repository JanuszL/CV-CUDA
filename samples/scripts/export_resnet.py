# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torchvision import models

# Get pretrained model
resnet50 = models.resnet50(pretrained=True)

# Export the model to ONNX
inputWidth = 224
inputHeight = 224
maxBatchSize = 32
x = torch.randn(maxBatchSize, 3, inputHeight, inputWidth, requires_grad=True)
torch.onnx.export(
    resnet50,
    x,
    "./models/resnet50.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "maxBatchSize"}, "output": {0: "maxBatchSize"}},
)
