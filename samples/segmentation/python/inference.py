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

"""
Semantic Segmentation Python sample

The semantic segmentation sample uses DeepLabv3 model from the torchvision
repository and shows the usage of CVCUDA by implementing a complete end-to-end
pipeline which can read images from the disk, pre-process them, run the inference
on them and save the overlay outputs back to the disk. This sample also gives an
overview of the interoperability of PyTorch and TensorRT with CVCUDA tensors and
operators.
"""

import os
import sys
import glob
import argparse
import torch
import torchnvjpeg
import torchvision.transforms.functional as F
from torchvision.models import segmentation as segmentation_models
import numpy as np
import nvcv
import tensorrt as trt

# Bring the commons folder from the samples directory into our path so that
# we can import modules from it.
common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)
from trt_utils import convert_onnx_to_tensorrt, setup_tensort_bindings  # noqa: E402


class SemanticSegmentationSample:
    def __init__(
        self,
        input_path,
        results_dir,
        visualization_class_name,
        batch_size,
        target_img_height,
        target_img_width,
        device_id,
        backend,
    ):
        self.input_path = input_path
        self.results_dir = results_dir
        self.visualization_class_name = visualization_class_name
        self.batch_size = batch_size
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.device_id = device_id
        self.backend = backend
        self.class_to_idx_dict = None

        if self.backend not in ["pytorch", "tensorrt"]:
            print(
                "Invalid backend option specified: %s. "
                "Currently supports: pytorch, tensorrt" % self.backend
            )
            exit(1)

        # Start by parsing the input_path expression first.
        if os.path.isfile(self.input_path):
            # Read the input image file.
            self.file_names = [self.input_path] * self.batch_size
            # Then create a dummy list with the data from the same file to simulate a
            # batch.
            self.data = [open(path, "rb").read() for path in self.file_names]

        elif os.path.isdir(self.input_path):
            # It is a directory. Grab all the images from it.
            self.file_names = glob.glob(os.path.join(self.input_path, "*.jpg"))
            self.data = [open(path, "rb").read() for path in self.file_names]
            print("Read a total of %d JPEG images." % len(self.data))

        else:
            print(
                "Input path not found. "
                "It is neither a valid JPEG file nor a directory: %s" % self.input_path
            )
            exit(1)

        if not os.path.isdir(self.results_dir):
            print("Output directory not found: %s" % self.results_dir)
            exit(1)

        if self.batch_size <= 0:
            print("batch_size must be a value >=1.")
            exit(1)

        if self.target_img_height < 10:
            print("target_img_height must be a value >=10.")
            exit(1)

        if self.target_img_width < 10:
            print("target_img_width must be a value >=10.")
            exit(1)

    def setup_model(self):
        # Setup the model and a few more things depending on the type of backend.
        if self.backend == "pytorch":
            # Fetch the segmentation index to class name information from the weights
            # meta properties.
            torch_model = segmentation_models.deeplabv3_resnet101
            weights_info = segmentation_models.DeepLabV3_ResNet101_Weights

            weights = weights_info.DEFAULT
            self.class_to_idx_dict = {
                cls: idx for (idx, cls) in enumerate(weights.meta["categories"])
            }

            if self.visualization_class_name not in self.class_to_idx_dict:
                print(
                    "Requested segmentation class '%s' is not supported by the "
                    "DeepLabV3 model." % self.visualization_class_name
                )
                print(
                    "All supported class names are: %s"
                    % ", ".join(self.class_to_idx_dict.keys())
                )
                exit(1)

            # Inference uses PyTorch to run a segmentation model on the pre-processed
            # input and outputs the segmentation masks.
            model = torch_model(weights=weights)
            model.cuda(self.device_id)
            model.eval()

            return model

        elif self.backend == "tensorrt":
            # For TensorRT, the process is the following:
            # We check if there already exists a TensorRT engine generated
            # previously. If not, we check if there exists an ONNX model generated
            # previously. If not, we will generate both of the one by one
            # and then use those.
            # The underlying pytorch model that we use in case of TensorRT
            # inference is the FCN model from torchvision. It is only used during
            # the conversion process and not during the inference.
            onnx_file_path = os.path.join(self.results_dir, "model.onnx")
            trt_engine_file_path = os.path.join(
                self.results_dir, "model.%d.trtmodel" % self.batch_size
            )

            torch_model = segmentation_models.fcn_resnet101
            weights_info = segmentation_models.FCN_ResNet101_Weights

            weights = weights_info.DEFAULT
            self.class_to_idx_dict = {
                cls: idx for (idx, cls) in enumerate(weights.meta["categories"])
            }

            if self.visualization_class_name not in self.class_to_idx_dict:
                print(
                    "Requested segmentation class '%s' is not supported by the "
                    "FCN model." % self.visualization_class_name
                )
                print(
                    "All supported class names are: %s"
                    % ", ".join(self.class_to_idx_dict.keys())
                )
                exit(1)

            # Check if we have a previously generated model.
            if not os.path.isfile(trt_engine_file_path):
                if not os.path.isfile(onnx_file_path):
                    # First we use PyTorch to create a segmentation model.
                    pyt_model = torch_model(weights=weights)
                    pyt_model.to("cuda")
                    pyt_model.eval()

                    # Allocate a dummy input to help generate an ONNX model.
                    dummy_x_in = torch.randn(
                        self.batch_size,
                        3,
                        self.target_img_height,
                        self.target_img_width,
                        requires_grad=False,
                    ).cuda()

                    # Generate an ONNX model using the PyTorch's onnx export.
                    torch.onnx.export(
                        pyt_model,
                        args=dummy_x_in,
                        f=onnx_file_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={
                            "input": {0: "batch_size"},
                            "output": {0: "batch_size"},
                        },
                    )

                    print("Generated an ONNX model and saved at: %s" % onnx_file_path)
                else:
                    print("Using a pre-built ONNX model from: %s" % onnx_file_path)

                # Now that we have an ONNX model, we will continue generating a
                # serialized TensorRT engine from it.
                success = convert_onnx_to_tensorrt(
                    onnx_file_path,
                    trt_engine_file_path,
                    max_batch_size=self.batch_size,
                    max_workspace_size=1,
                )
                if success:
                    print("Generated TensorRT engine in: %s" % trt_engine_file_path)
                else:
                    print("Failed to generate the TensorRT engine.")
                    exit(1)

            else:
                print(
                    "Using a pre-built TensorRT engine from: %s" % trt_engine_file_path
                )

            # Once the TensorRT engine generation is all done, we load it.
            trt_logger = trt.Logger(trt.Logger.INFO)
            with open(trt_engine_file_path, "rb") as f, trt.Runtime(
                trt_logger
            ) as runtime:
                trt_model = runtime.deserialize_cuda_engine(f.read())

            # Create execution context.
            context = trt_model.create_execution_context()

            # Allocate the output bindings.
            output_tensors, output_idx = setup_tensort_bindings(
                trt_model, self.device_id
            )

            return context, output_tensors, output_idx

        else:
            print(
                "Invalid backend option specified: %s. "
                "Currently supports: pytorch, tensorrt" % self.backend
            )
            exit(1)

    def execute_inference(self, model_info, torch_preprocessed_tensor):
        # Executes inference depending on the type of the backend.
        if self.backend == "pytorch":
            with torch.no_grad():
                infer_output = model_info(torch_preprocessed_tensor)["out"]

            return infer_output

        elif self.backend == "tensorrt":
            # Setup TensorRT IO binding pointers.
            context, output_tensors, output_idx = model_info  # Un-pack this.

            # We need to check the allocated batch size and the required batch
            # size. Sometimes, during to batching, the last batch may be of
            # less size than the batch size. In those cases, we would simply
            # pad that with zero inputs and discard its output later on.
            allocated_batch_size = output_tensors[output_idx].shape[0]
            required_batch_size = torch_preprocessed_tensor.shape[0]

            if allocated_batch_size != required_batch_size:
                # Need to pad the input with extra zeros tensors.
                new_input_shape = [allocated_batch_size - required_batch_size] + list(
                    torch_preprocessed_tensor.shape[1:]
                )

                # Allocate just the extra input required.
                extra_input = torch.zeros(
                    size=new_input_shape,
                    dtype=torch_preprocessed_tensor.dtype,
                    device=self.device_id,
                )

                # Update the existing input tensor by joining it with the newly
                # created input.
                torch_preprocessed_tensor = torch.cat(
                    (torch_preprocessed_tensor, extra_input)
                )

            # Prepare the TensorRT I/O bindings.
            input_bindings = [torch_preprocessed_tensor.data_ptr()]
            output_bindings = []
            for t in output_tensors:
                output_bindings.append(t.data_ptr())
            io_bindings = input_bindings + output_bindings

            # Execute synchronously.
            context.execute_v2(bindings=io_bindings)
            infer_output = output_tensors[output_idx]

            # Finally, check if we had padded the input. If so, we need to
            # discard the extra output.
            if allocated_batch_size != required_batch_size:
                # We need remove the padded output.
                infer_output = torch.split(
                    infer_output,
                    [required_batch_size, allocated_batch_size - required_batch_size],
                )[0]

            return infer_output

        else:
            print(
                "Invalid backend option specified: %s. "
                "Currently supports: pytorch, tensorrt" % self.backend
            )
            exit(1)

    def run(self):
        # Runs the complete sample end-to-end.
        max_image_size = 1024 * 1024 * 3  # Maximum possible image size.

        # First setup the model.
        model_info = self.setup_model()

        # Next, we would batchify the file_list and data_list based on the
        # batch size and start processing these batches one by one.
        file_name_batches = [
            self.file_names[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.file_names), self.batch_size)
        ]
        data_batches = [
            self.data[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.data), self.batch_size)
        ]
        batch_idx = 0

        # We will use the torchnvjpeg based decoder on the GPU. This will be
        # allocated once during the first run or whenever a batch size change
        # happens.
        decoder = None

        for file_name_batch, data_batch in zip(file_name_batches, data_batches):
            print("Processing batch %d of %d" % (batch_idx + 1, len(file_name_batches)))
            effective_batch_size = len(file_name_batch)

            # Decode in batch using torchnvjpeg decoder on the GPU.
            if not decoder or effective_batch_size != self.batch_size:
                decoder = torchnvjpeg.Decoder(
                    0,
                    0,
                    True,
                    self.device_id,
                    effective_batch_size,
                    8,  # this is max_cpu_threads parameter. Not used internally.
                    max_image_size,
                    torch.cuda.current_stream(self.device_id),
                )
            image_tensor_list = decoder.batch_decode(data_batch)

            # Convert the list of tensors to a tensor itself.
            image_tensors = torch.stack(image_tensor_list)

            # Also save an NCHW version of the image tensors.
            image_tensors_nchw = image_tensors.permute(0, 3, 1, 2)  # from NHWC to NCHW

            input_image_height, input_image_width = (
                image_tensors.shape[1],
                image_tensors.shape[2],
            )

            # A torch tensor can be wrapped into a CVCUDA Object using the "as_tensor"
            # function in the specified layout. The datatype and dimensions are derived
            # directly from the torch tensor.
            nvcv_input_tensor = nvcv.as_tensor(image_tensors, "NHWC")

            # Start the pre-processing now. For segmentation, pre-processing includes
            # the following sequence of operations.
            # Resize -> DataType Convert(U8->F32) -> Normalize -> Interleaved to Planar

            # Resize to the input network dimensions.
            nvcv_resized_tensor = nvcv_input_tensor.resize(
                (
                    effective_batch_size,
                    self.target_img_height,
                    self.target_img_width,
                    3,
                ),
                nvcv.Interp.LINEAR,
            )

            # Convert to the data type and range of values needed by the input layer
            # i.e uint8->float. The values are first scaled to the 0-1 range.
            nvcv_float_tensor = nvcv_resized_tensor.convertto(np.float32, scale=1 / 255)

            # Normalize using mean and std-dev
            mean_tensor = torch.Tensor([0.485, 0.456, 0.406])
            stddev_tensor = torch.Tensor([0.229, 0.224, 0.225])
            mean_tensor = mean_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
            stddev_tensor = stddev_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
            nvcv_mean_tensor = nvcv.as_tensor(mean_tensor, "NHWC")
            nvcv_stddev_tensor = nvcv.as_tensor(stddev_tensor, "NHWC")
            nvcv_normalized_tensor = nvcv_float_tensor.normalize(
                base=nvcv_mean_tensor,
                scale=nvcv_stddev_tensor,
                flags=nvcv.NormalizeFlags.SCALE_IS_STDDEV,
            )

            # The final stage in the pre-process pipeline includes converting the NHWC
            # buffer into a NCHW buffer.
            nvcv_preprocessed_tensor = nvcv_normalized_tensor.reformat("NCHW")

            # Execute the inference after converting the tensor back to Torch.
            torch_preprocessed_tensor = torch.as_tensor(
                nvcv_preprocessed_tensor.cuda(),
                device=torch.device("cuda", self.device_id),
            )
            infer_output = self.execute_inference(model_info, torch_preprocessed_tensor)

            # Once the inference is over we would start the post-processing steps.
            # First, we normalize the probability scores from the network.
            normalized_masks = torch.nn.functional.softmax(infer_output, dim=1)

            # Then filter based on the scores corresponding only to the class of
            # interest
            class_masks = (
                normalized_masks.argmax(dim=1)
                == self.class_to_idx_dict[self.visualization_class_name]
            )
            class_masks = torch.unsqueeze(class_masks, dim=-1)  # Makes it NHWC
            class_masks = class_masks.type(torch.uint8)  # Make it uint8 from bool

            # Then convert the masks back to CV-CUDA tensor for rest of the
            # post-processing:
            # 1) Up-scaling back to the original image dimensions
            # 2) Apply blur on the original images and overlay on the original image.

            # Convert back to CV-CUDA tensor
            nvcv_class_masks = nvcv.as_tensor(class_masks.cuda(), "NHWC")
            # Upscale it.
            nvcv_class_masks_upscaled = nvcv_class_masks.resize(
                (effective_batch_size, input_image_height, input_image_width, 1),
                nvcv.Interp.LINEAR,
            )
            # Convert back to PyTorch.
            class_masks_upscaled = torch.as_tensor(
                nvcv_class_masks_upscaled.cuda(),
                device=torch.device("cuda", self.device_id),
            )
            # Repeat in last dimension to make the mask 3 channel
            class_masks_upscaled = class_masks_upscaled.repeat(1, 1, 1, 3)
            class_masks_upscaled_nchw = class_masks_upscaled.permute(
                0, 3, 1, 2
            )  # from NHWC to NCHW

            # Blur the input images using the median blur op and convert to PyTorch.
            nvcv_blurred_input_imgs = nvcv_input_tensor.median_blur(ksize=(27, 27))
            nvcv_blurred_input_imgs = nvcv_blurred_input_imgs.reformat("NCHW")
            blurred_input_imgs = torch.as_tensor(
                nvcv_blurred_input_imgs.cuda(),
                device=torch.device("cuda", self.device_id),
            )

            # Create an overlay image. We do this by selectively blurring out pixels
            # in the input image where the class mask prediction was absent (i.e. False)
            # We already have all the things required for this: The input images,
            # the blurred version of the input images and the upscale version
            # of the mask
            mask_absent = class_masks_upscaled_nchw == 0
            image_tensors_nchw[mask_absent] = blurred_input_imgs[
                mask_absent
            ]  # In-place

            # Loop over all the images in the current batch and save the
            # inference results.
            for img_idx in range(effective_batch_size):
                img_name = os.path.splitext(os.path.basename(file_name_batch[img_idx]))[
                    0
                ]
                results_path = os.path.join(self.results_dir, "out_%s.jpg" % img_name)
                print(
                    "\tSaving the overlay result for %s class for to: %s"
                    % (self.visualization_class_name, results_path)
                )

                # Convert the overlay which was in-place saved in
                # image_tensors_nchw to a PIL image on the CPU and save it.
                overlay_cpu = image_tensors_nchw[img_idx].detach().cpu()
                overlay_pil = F.to_pil_image(overlay_cpu)
                overlay_pil.save(results_path)

            # Increment the batch counter.
            batch_idx += 1


def main():
    parser = argparse.ArgumentParser(
        "Semantic segmentation sample using CV-CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="./assets/Weimaraner.jpg",
        type=str,
        help="Either a path to a JPEG image or a directory containing JPEG "
        "images to use as input.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/tmp",
        type=str,
        help="The folder where the output segmentation overlay should be stored.",
    )

    parser.add_argument(
        "-c",
        "--class_name",
        default="dog",
        type=str,
        help="The segmentation class to visualize the results for.",
    )

    parser.add_argument(
        "-th",
        "--target_img_height",
        default=520,
        type=int,
        help="The height to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-tw",
        "--target_img_width",
        default=520,
        type=int,
        help="The width to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="Artificially simulated batch size. The same input image will be read and "
        "used this many times. Useful for performance bench-marking.",
    )

    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="The GPU device to use for this sample.",
    )

    parser.add_argument(
        "-bk",
        "--backend",
        default="pytorch",
        type=str,
        help="The inference backend to use. Currently supports pytorch, tensorrt.",
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    # Run the sample.
    sample = SemanticSegmentationSample(
        args.input_path,
        args.output_dir,
        args.class_name,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
        args.backend,
    )

    sample.run()


if __name__ == "__main__":
    main()
