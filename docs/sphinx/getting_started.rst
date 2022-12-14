..
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
