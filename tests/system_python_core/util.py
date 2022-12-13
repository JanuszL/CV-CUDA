# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nvcv
import numpy as np
import numbers
import torch


DTYPE = {
    nvcv.Format.RGB8: np.uint8,
    nvcv.Format.RGBf32: np.float32,
    nvcv.Format.F32: np.float32,
    nvcv.Format.U8: np.uint8,
    nvcv.Format.S16: np.int16,
    nvcv.Format.S32: np.int32,
}


def create_tensor(shape, dtype, layout, max_random, odd_only=False):
    """Create a tensor with shape, (numpy) dtype, layout (e.g. NC, HWC, NHWC),
    max_random as the maximum random value (exclusive) inside the tensor, and
    odd_only to have only odd values inside the tensor (max_random becomes inclusive)
    """
    if type(max_random) is tuple or type(max_random) is list:
        assert len(max_random) == shape[-1]
    if issubclass(dtype, numbers.Integral):
        h_data = np.random.randint(max_random, size=shape)
        if odd_only:
            make_odd = np.vectorize(lambda x: x if x % 2 == 1 else x + 1)
            h_data = make_odd(h_data)
    elif issubclass(dtype, numbers.Real):
        h_data = np.random.random_sample(shape) * np.array(max_random)
    h_data = h_data.astype(dtype)
    d_data = torch.from_numpy(h_data).cuda()
    tensor = nvcv.as_tensor(d_data, layout=layout)
    return tensor


def create_image(size, img_format, max_random):
    """Create an image with size, (nvcv) img_format, and
    max_random as the maximum random value (exclusive) inside the image
    """
    h_data = np.random.rand(size[1], size[0], img_format.channels) * max_random
    h_data = h_data.astype(DTYPE[img_format])
    d_data = torch.from_numpy(h_data).cuda()
    image = nvcv.as_image(d_data)
    return image


def create_image_batch(
    num_images, img_format, size=(0, 0), max_size=(128, 128), max_random=1
):
    """Create an image batch with num_images, (nvcv) img_format, size (can be zero
    to use random) or max_size, as a random size in [1, max_size), and
    max_random as the maximum random value (exclusive) inside each image
    """
    image_batch = nvcv.ImageBatchVarShape(num_images)
    for i in range(num_images):
        w = size[0] if size[0] > 0 else np.random.randint(1, max_size[0])
        h = size[1] if size[1] > 0 else np.random.randint(1, max_size[1])
        image_batch.pushback(create_image((w, h), img_format, max_random))
    return image_batch


def clone_image_batch(input_image_batch):
    """Clone an image batch given as input"""
    output_image_batch = nvcv.ImageBatchVarShape(input_image_batch.capacity)
    for input_image in input_image_batch:
        image = nvcv.Image(input_image.size, input_image.format)
        output_image_batch.pushback(image)
    return output_image_batch
