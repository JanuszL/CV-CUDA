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

.. _python_classification:

Image Classification
====================

In this example we will cover use of CVCUDA to accalerate the preprocessing pipeline in DL inference usecase.
The preprocessing pipeline converts the input image to the required format for the input layer of the model.
We will use the Resnet50 model pretrained on Imagenet to implement an image classification pipeline.

The preprocesing operations required for Resnet50 include:

Resize -> Convert Datatype(Float) -> Normalize (std deviation/mean) -> Interleaved to planar

Writing the Sample App
----------------------

The first stage in the sample pipeline is loading the Input image

We will use NvJpeg library to decode the images into the desired color format and create a buffer on the device

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :linenos:
   :lines: 37-47

Once the device buffer is created we will wrap the externally allocated buffer in a CVCUDA Tensor

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 49-52

We will convert the input tensor to interleaved format for the rest of the preprocessing operations

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 56-59

The input buffer is now ready for the preprocessing stage

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 61-110

The preprocessed tensor is used as an input to the resnet model for inference. The cvcuda tensor
can be exported to torch using the .cuda() operator. If the device type of the torch tensor and
cvcuda tensor are same there will be no memory copy

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 112-121

The final stage in the pipeline is the post processing to apply softmax to normalize the score and sort the scores to get the TopN scores

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 123-144

Running the Sample
------------------
Set the image path, labels path and batch size in the python script

.. code-block:: bash

   python3 sample/classification/python/inference.py

Sample Output
-------------

The top 5 classification results for the tabby_cat_tiger.jpg image is as follows:

.. code-block:: bash

   Class :  tiger cat  Score :  0.7251133322715759
   Class :  tabby, tabby cat  Score :  0.15487350523471832
   Class :  Egyptian cat  Score :  0.08538217097520828
   Class :  lynx, catamount  Score :  0.020933201536536217
   Class :  leopard, Panthera pardus  Score :  0.002835722640156746
