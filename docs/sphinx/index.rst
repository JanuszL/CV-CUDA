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
