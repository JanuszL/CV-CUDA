# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvcv
import numpy as np
import numbers
import torch
import copy


DTYPE = {
    nvcv.Format.RGB8: np.uint8,
    nvcv.Format.RGBf32: np.float32,
    nvcv.Format.F32: np.float32,
    nvcv.Format.U8: np.uint8,
    nvcv.Format.U16: np.uint16,
    nvcv.Format.S16: np.int16,
    nvcv.Format.S32: np.int32,
}


def to_cuda_buffer(host):
    orig_dtype = copy.copy(host.dtype)

    # torch doesn't accept uint16. Let's make it believe
    # it is handling int16 instead.
    if host.dtype == np.uint16:
        host.dtype = np.int16

    dev = torch.as_tensor(host, device="cuda").cuda()
    host.dtype = orig_dtype  # restore it

    class CudaBuffer:
        __cuda_array_interface = None
        obj = None

    # The cuda buffer only needs the cuda array interface.
    # We can then set its dtype to whatever we want.
    buf = CudaBuffer()
    buf.__cuda_array_interface__ = dev.__cuda_array_interface__
    buf.__cuda_array_interface__["typestr"] = orig_dtype.str
    buf.obj = dev  # make sure it holds a reference to the torch buffer

    return buf


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
    tensor = nvcv.as_tensor(to_cuda_buffer(h_data), layout=layout)
    return tensor


def create_image(size, img_format, max_random):
    """Create an image with size, (nvcv) img_format, and
    max_random as the maximum random value (exclusive) inside the image
    """
    h_data = np.random.rand(size[1], size[0], img_format.channels) * max_random
    h_data = h_data.astype(DTYPE[img_format])

    image = nvcv.as_image(to_cuda_buffer(h_data))
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
