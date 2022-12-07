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
import argparse

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


def classification(
    image_file,
    labels_file,
    batch_size,
    layer_height,
    layer_width,
):

    # Read the input imagea file
    data = read_file(image_file)  # raw data is on CPU

    # NvJpeg can be used to decode the image
    # to the necessary color format on the device
    input_image = decode_jpeg(data, device="cuda")  # decoded image in on GPU

    # decode_jpeg is currently limited to batch_size 1
    # and Planar format (CHW)

    # A torch tensor can be wrapped into a CVCUDA Object using the "as_tensor"
    # function in the specified layout. The datatype and dimensions are derived
    # directly from the torch tensor.
    nvcv_input_tensor = nvcv.as_tensor(input_image.cuda(), "CHW")

    # The input image is now ready to be used

    # The Reformat operator can be used to convert CHW format to NHWC
    # for the rest of the preprocessing operations

    nvcv_interleaved_tensor = nvcv_input_tensor.reformat("NHWC")

    """
    Preprocessing includes the following sequence of operations.
    Resize -> DataType Convert(U8->F32) -> Normalize(Apply mean and std deviation)
    -> Interleaved to Planar
    """

    # Resize
    # Resize to the input network dimensions
    nvcv_resize_tensor = nvcv_interleaved_tensor.resize(
        (batch_size, layer_width, layer_height, 3), nvcv.Interp.CUBIC
    )

    # Convert to the data type and range of values needed by the input layer
    # i.e uint8->float. A Scale is applied to normalize the values in the range 0-1
    nvcv_convert_tensor = nvcv_resize_tensor.convertto(np.float32, scale=1 / 255)

    """
    The input to the network needs to be normalized based on the mean and
    std deviation value to standardize the input data.
    """

    # Create a torch tensor to store the mean and standard deviation values for R,G,B
    scale = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale_tensor = torch.Tensor(scale)
    stddev_tensor = torch.Tensor(std)

    # Reshape the the number of channels. The R,G,B values scale and offset will be
    # applied to every color plane respectively across the batch
    scale_tensor = torch.reshape(scale_tensor, (1, 1, 1, 3)).cuda()
    stddev_tensor = torch.reshape(stddev_tensor, (1, 1, 1, 3)).cuda()

    # Wrap the torch tensor in a CVCUDA Tensor
    nvcv_scale_tensor = nvcv.as_tensor(scale_tensor, "NHWC")
    nvcv_base_tensor = nvcv.as_tensor(stddev_tensor, "NHWC")

    # Apply the normalize operator and indicate the scale values are std deviation
    # i.e scale = 1/stddev
    nvcv_norm_tensor = nvcv_convert_tensor.normalize(
        nvcv_base_tensor, nvcv_scale_tensor, nvcv.NormalizeFlags.SCALE_IS_STDDEV
    )

    # The final stage in the preprocess pipeline includes converting the RGB buffer
    # into a planar buffer
    nvcv_preprocessed_tensor = nvcv_norm_tensor.reformat("NCHW")

    # Inference uses pytorch to run a resnet50 model on the preprocessed
    # input and outputs the classification scores for 1000 classes
    # Load Resnet model pretrained on Imagenet
    resnet50 = models.resnet50(pretrained=True)
    resnet50.to("cuda")
    resnet50.eval()

    # Run inference on the preprocessed input
    torch_preprocessed_tensor = torch.as_tensor(
        nvcv_preprocessed_tensor.cuda(), device="cuda"
    )
    infer_output = resnet50(torch_preprocessed_tensor)

    """
    Postprocessing function normalizes the classification score from the network
    and sorts the scores to get the TopN classification scores.
    """
    # top results to print out
    topN = 5

    # Read and parse the classes
    with open(labels_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Apply softmax to Normalize scores between 0-1
    scores = torch.nn.functional.softmax(infer_output, dim=1)[0]

    # Sort output scores in descending order
    _, indices = torch.sort(infer_output, descending=True)

    # Display Top N Results
    [
        print("Class : ", classes[idx], " Score : ", scores[idx].item())
        for idx in indices[0][:topN]
    ]


def main():
    parser = argparse.ArgumentParser(
        "Classification sample using CV-CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_image",
        default="tabby_tiger_cat.jpg",
        type=str,
        help="The input image to read.",
    )

    parser.add_argument(
        "-l",
        "--labels_file",
        default="imagenet-classes.txt",
        type=str,
        help="The labels file to read and parse.",
    )

    parser.add_argument(
        "-th",
        "--target_img_height",
        default=224,
        type=int,
        help="The height to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-tw",
        "--target_img_width",
        default=224,
        type=int,
        help="The width to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="Input Batch size"
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    # Run the sample.
    classification(
        args.input_image,
        args.labels_file,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
    )


if __name__ == "__main__":
    main()
