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

.. _cpp_cropresize:

Crop And Resize
===============

In this example we will cover some basic concepts to show how to use the CVCUDA C++ API which includes usage of Tensor
,wrapping externally allocated data in CVCUDA Tensor and using Tensors with operators.

Creating a CMake Project
------------------------

Create the cmake project to build the application as follows. The <samples/common> folder provides utilities common across the C++ samples including IO utilities to read and write images using NvJpeg.

.. literalinclude:: ../../../samples/cropandresize/CMakeLists.txt
   :language: cpp
   :lines: 15-19

Writing the Sample App
----------------------

The first stage in the sample pipeline is loading the Input image.
A cuda stream is created to enqueue all the tasks

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 114-115

We will use NvJpeg library to decode the images into the required color format and create a buffer on the device.
Since we need a contiguous buffer for a batch, we will preallocate the Tensor buffer for the input batch.

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 117-126

The Tensor Buffer is then wrapped to create a Tensor Object for which we will calculate the requirements of the buffer such as pitch bytes and alignment

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 128-138

The CVCUDA Tensor is now ready to be used by the operators.

We will allocate the Tensors required for Resize and Crop using CVCUDA Allocator.

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 165-168

Initialize the resize and crop operators

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 176-179

We can now enqueue both the operations in the stream

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 181-185

To access the output we will synchronize the stream and copy to the CPU Output buffer
We will use the utility below to sync and write the CPU output buffer into a bitmap file

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 195-196

Destroy the cuda stream created

.. literalinclude:: ../../../samples/cropandresize/Main.cpp
   :language: cpp
   :lines: 198-199

Build and Run the Sample
------------------------

The sample can now be compiled using cmake.

.. code-block:: bash

   mkdir build
   cd build
   cmake .. && make

To run the sample

.. code-block:: bash

   ./build/nvcv_samples_cropandresize -i <image path> -b <batch size>

Sample Output
-------------

Input Image of size 700x700

.. image:: ../../../samples/assets/tabby_tiger_cat.jpg
   :width: 350

Output Image cropped with ROI [150, 50, 400, 300] and resized to 320x240

.. image:: tabby_cat_crop.bmp
   :width: 160
