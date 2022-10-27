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

The Tensor can also be allocated using CVCUDA. We will convert the input tensor to interleaved format for
the rest of the preprocessing operations

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 56-65

The input buffer is now ready for the preprocessing stage

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 67-130

The preprocessed tensor is used as an input to the resnet model for inference

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 132-140

The final stage in the pipeline is the post processing to apply softmax to normalize the score and sort the scores to get the TopN scores

.. literalinclude:: ../../../samples/classification/python/inference.py
   :language: python
   :lines: 149-163

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
