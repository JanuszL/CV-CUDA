..
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

.. _prealpha:

PreAlpha
========

CV-CUDA-0.1.0 is the first release of CV-CUDA. This release is for evaluation purposes only.

Release Highlights
------------------

This CV-CUDA release includes the following key features:

* Core components including Image and Tensor with Batch support
* Utilities to help write CUDA kernels
* 6 Operators - Reformat, Resize, Custom Crop, Normalize, PadAndStack, ConvertTo
* Tensor interoperability with pytorch/gpu, Image interoperability with pytorch/gpu, pillow/cpu, opencv/cpu
* Python bindings
* Sample applications
* API documentation

Compatibility
-------------
This section highlights the compute stack CV-CUDA has been tested on

* Ubuntu x86 >= 20.04
* CUDA driver >= 11.7

The Sample applications based on TensorRT have been tested with TensorRT >= 8.5

Known Issues
------------
* There will be few updates in the Tensor API, Image Formats and Operator names in the next release
* Limitations in the usage of the operators which are described in the API documentation

License
-------
Nvidia Software Evaluation License
