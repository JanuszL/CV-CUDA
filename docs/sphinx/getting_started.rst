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

.. _getting_started:

Getting Started
===============

This section provides a step-by-step procedure for building a simple pipeline that demonstrates core CVCUDA programming concepts.

Prerequisites
-------------

This section describes the recommended dependencies to compile cvcuda

* Ubuntu >= 20.04
* CUDA driver >= 11.7

C++ Sample Dependencies:

* TensorRT to run the Image classification sample pipeline

Python Sample Dependencies:

* Torch
* Torchvision

Refer to the :ref:`Installation` docs for the sample installation guide using *.deb or .tar installers.
Refer to the sample README for instructions to compile samples from the source.

Tutorial
--------

There are three ways to use CVCUDA:

* C API
* C++ API
* Python API.

The following section provides tutorials in C++ and Python.

.. toctree::
    :maxdepth: 1
    :caption: Tutorial
    :hidden:

    C++ API <samples/cpp_cropresize>
    Python API <samples/python_classification>
