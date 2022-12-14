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

.. _cvcuda_doc_system:

CV-CUDA
============================

Project CV-CUDA is NVIDIA's open-source, graphics processing unit (GPU)-accelerated toolkit for computer vision (CV).

The toolkit provides

*  a unified, core set of highly performant CV kernels as standalone operators to build efficient pipelines
*  batching support, with variable shape images in one batch
*  utilities to write CUDA kernels for CV algorithms.

CV CUDA Algorithms
------------------

The following CV algorithms are available:

*       CustomCrop
*       Normalize
*       Padding
*       Reformat
*       Resize

Read on to learn more about how to use and benefit from CV-CUDA.

.. toctree::
    :caption: Beginner's Guide
    :maxdepth: 1
    :hidden:

    Installation <installation>
    Getting Started <getting_started>

.. toctree::
    :caption: API Documentation
    :maxdepth: 2
    :hidden:

    C Modules <modules/c_modules>
    C++ Modules <modules/cpp_modules>
    Index <_exhale_api/cvcuda_api>

.. toctree::
    :caption: Benchmarks
    :maxdepth: 1
    :hidden:

    Performance Benchmark <perf_benchmark>

.. toctree::
    :caption: Release Notes
    :maxdepth: 1
    :hidden:

    PreAlpha <relnotes/prealpha>
