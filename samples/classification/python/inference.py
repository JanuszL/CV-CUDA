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
from torchvision.io.image import read_file, decode_jpeg
import numpy as np

# Import CVCUDA module
import nvcv

"""
Image Classification python sample

The image classification sample uses Resnet50 based model trained on Imagenet
The sample app pipeline includes preprocessing, inference and post process stages
which takes as input a batch of images and returns the TopN classification results
of each image.

This sample gives an overview of the interoperability of pytorch with CVCUDA
tensors and operators
"""

# Set the image and labels file
filename = "./assets/tabby_tiger_cat.jpg"
labelsfile = "./models/imagenet-classes.txt"

# Read the input imagea file
data = read_file(filename)  # raw data is on CPU

# NvJpeg can be used to decode the image
# to the necessary color format on the device
inputImage = decode_jpeg(data, device="cuda")  # decoded image in on GPU
imageWidth = inputImage.shape[2]
imageHeight = inputImage.shape[1]

# decode_jpeg is currently limited to batchSize 1
# and Planar format (CHW)

# A torch tensor can be wrapped into a CVCUDA Object using the "as_tensor"
# function in the specified layout. The datatype and dimensions are derived
# directly from the torch tensor.
nvcvnInputTensor = nvcv.as_tensor(inputImage, "CHW")

# The input image is now ready to be used

# A Tensor can also be created by using the CVCUDA memory allocator as below
# by specifying the dimensions, datatype and layout
nvcvInterleavedTensor = nvcv.Tensor([1, imageHeight, imageWidth, 3], np.uint8, "NHWC")

# nvcvnInterleavedTensor is a CVCUDA Tensor allocated on the GPU

# The Reformat operator can be used to convert CHW format to NHWC
# for the rest of the preprocessing operations

tmp = nvcvnInputTensor.reformat_into(nvcvInterleavedTensor)

"""
Preprocessing includes the following sequence of operations.
Resize -> DataType COnvert(U8->F32) -> Normalize(Apply mean and std deviation)
-> Interleaved to Planar
"""

# Model settings
layerHeight = 224
layerWidth = 224
batchSize = 1

# Resize
# Create a tensor for the resize output based on layer dimensions
nvcvResizeTensor = nvcv.Tensor(
    [batchSize, layerHeight, layerWidth, 3], np.uint8, "NHWC"
)
# Resize to the input network dimensions
tmp = nvcvInterleavedTensor.resize_into(nvcvResizeTensor, nvcv.Interp.CUBIC)

# Convert to the data type and range of values needed by the input layer
# i.e uint8->float. A Scale is applied to normalize the values in the range 0-1
scale = 1 / 255
offset = 0
nvcvConvertTensor = nvcv.Tensor(
    [batchSize, layerHeight, layerWidth, 3], np.float32, "NHWC"
)
tmp = nvcvResizeTensor.convertto_into(nvcvConvertTensor, scale, offset)

"""
The input to the network needs to be normalized based on the mean and
std deviation value to standardize the input data.
"""

# Create a torch tensor to store the mean and standard deviation values for R,G,B
scale = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
scaleTensor = torch.Tensor(scale)
stdTensor = torch.Tensor(std)

# Reshape the the number of channels. The R,G,B values scale and offset will be
# applied to every color plane respectively across the batch
scaleTensor = torch.reshape(scaleTensor, (1, 1, 1, 3)).cuda()
stdTensor = torch.reshape(stdTensor, (1, 1, 1, 3)).cuda()

# Wrap the torch tensor in a CVCUDA Tensor
nvcvScaleTensor = nvcv.as_tensor(scaleTensor, "NHWC")
nvcvBaseTensor = nvcv.as_tensor(stdTensor, "NHWC")

# Apply the normalize operator and indicate the scale values are std deviation
# i.e scale = 1/stddev
nvcvNormTensor = nvcv.Tensor(
    [batchSize, layerHeight, layerWidth, 3], np.float32, "NHWC"
)
tmp = nvcvConvertTensor.normalize_into(
    nvcvNormTensor, nvcvBaseTensor, nvcvScaleTensor, nvcv.NormalizeFlags.SCALE_IS_STDDEV
)

# The final stage in the preprocess pipeline includes converting the RGB buffer
# into a planar buffer
preprocessedTensor = torch.zeros(
    [batchSize, 3, layerHeight, layerWidth], dtype=torch.float32, device="cuda"
)
nvcvPlanarTensor = nvcv.as_tensor(preprocessedTensor, "NCHW")
tmp = nvcvNormTensor.reformat_into(nvcvPlanarTensor)

# Inference uses pytorch to run a resnet50 model on the preprocessed input and outputs
# the classification scores for 1000 classes
# Load Resnet model pretrained on Imagenet
resnet50 = models.resnet50(pretrained=True)
resnet50.to("cuda")
resnet50.eval()

# Run inference on the preprocessed input
inferOutput = resnet50(preprocessedTensor)

"""
Postprocessing function normalizes the classification score from the network and sorts
the scores to get the TopN classification scores.
"""
# top results to print out
topN = 5

# Read and parse the classes
with open(labelsfile, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Apply softmax to Normalize scores between 0-1
scores = torch.nn.functional.softmax(inferOutput, dim=1)[0]

# Sort output scores in descending order
_, indices = torch.sort(inferOutput, descending=True)

# Display Top N Results
[
    print("Class : ", classes[idx], " Score : ", scores[idx].item())
    for idx in indices[0][:topN]
]
