#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Classification
# resnet50

mkdir -p models

if [ ! -f ./models/imagenet-classes.txt ]
then
        wget https://raw.githubusercontent.com/xmartlabs/caffeflow/master/examples/imagenet/imagenet-classes.txt -O models/imagenet-classes.txt
fi

if [ ! -f ./models/resnet50.engine ]
then
        /opt/tensorrt/bin/trtexec --onnx=models/resnet50.onnx --saveEngine=models/resnet50.engine --minShapes=input:1x3x224x224 --maxShapes=input:32x3x224x224 --optShapes=input:32x3x224x224
fi
