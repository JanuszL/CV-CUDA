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

# Usage: run_samples.sh

mkdir -p models

# Export onnx model from torch
python ./scripts/export_resnet.py

# Serialize models
./scripts/serialize_models.sh

# Batch size 1
LD_LIBRARY_PATH=./lib ./bin/nvcv_samples_classification -e ./models/resnet50.engine -i ./assets/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 1

# Batch size 2
LD_LIBRARY_PATH=./lib ./bin/nvcv_samples_classification -e ./models/resnet50.engine -i ./assets/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 2
