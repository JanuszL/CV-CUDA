# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import logging
import cvcuda
import torch
import nvtx


class PreprocessorCvcuda:
    # docs_tag: begin_init_preprocessorcvcuda
    def __init__(self, device_id):
        self.scale = 1 / 255
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id

        self.logger.info("Using CVCUDA as preprocessor.")
        # docs_tag: end_init_preprocessorcvcuda

    # docs_tag: begin_call_preprocessorcvcuda
    def __call__(self, frame_nhwc, out_size):
        nvtx.push_range("preprocess.cvcuda")

        # docs_tag: begin_tensor_conversion
        # Need to check what type of input we have received:
        # 1) CVCUDA tensor --> Nothing needs to be done.
        # 2) Numpy Array --> Convert to torch tensor first and then CVCUDA tensor
        # 3) Torch Tensor --> Convert to CVCUDA tensor
        if isinstance(frame_nhwc, torch.Tensor):
            frame_nhwc = cvcuda.as_tensor(frame_nhwc, "NHWC")
            has_copy = False
        elif isinstance(frame_nhwc, np.ndarray):
            has_copy = True  # noqa: F841
            frame_nhwc = cvcuda.as_tensor(
                torch.as_tensor(frame_nhwc).to(
                    device="cuda:%d" % self.device_id, non_blocking=True
                ),
                "NHWC",
            )
        # docs_tag: end_tensor_conversion

        # docs_tag: begin_preproc_pipeline
        # Resize the tensor to a different size.
        # NOTE: This resize is done after the data has been converted to a NHWC Tensor format
        #       That means the height and width of the frames/images are already same, unlike
        #       a python list of HWC tensors.
        #       This resize is only going to help it downscale to a fixed size and not
        #       to help resize images with different sizes to a fixed size. If you have a folder
        #       full of images with all different sizes, it would be best to run this sample with
        #       batch size of 1. That way, this resize operation will be able to resize all the images.
        resized = cvcuda.resize(
            frame_nhwc,
            (
                frame_nhwc.shape[0],
                out_size[1],
                out_size[0],
                frame_nhwc.shape[3],
            ),
            cvcuda.Interp.LINEAR,
        )

        # Convert to floating point range 0-1.
        normalized = cvcuda.convertto(resized, np.float32, scale=1 / 255)

        # Convert it to NCHW layout and return it.
        normalized = cvcuda.reformat(normalized, "NCHW")

        nvtx.pop_range()

        # Return 3 pieces of information:
        #   1. The original nhwc frame
        #   2. The resized frame
        #   3. The normalized frame.
        return (
            frame_nhwc,
            resized,
            normalized,
        )
        # docs_tag: end_preproc_pipeline


class PostprocessorCvcuda:
    def __init__(self, threshold, device_id, output_layout, gpu_output):
        # docs_tag: begin_init_postprocessorcvcuda
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id

        self.gpu_output = gpu_output
        self.output_layout = output_layout

        # confidence threshold of the detections
        self.threshold = threshold
        # Peoplenet model uses Gridbox system which divides an input image into a grid and
        # predicts four normalized bounding-box parameters for each grid.
        # The number of grid boxes is determined by the model architecture.
        # For peoplenet model, the 960x544 input image is divided into 60x34 grids.
        self.num_rows = 34
        self.num_cols = 60
        self.bbox_norm = 35
        self.offset = 0.5
        # Number of classes the mode is trained on
        self.num_classes = 3

        # Define the Bounding Box utils
        self.bboxutil = BoundingBoxUtilsCvcuda()

        self.logger.info("Using CVCUDA as post-processor.")

    # docs_tag: end_init_postprocessorcvcuda

    def interpolate_bboxes(
        self,
        curr_left,
        curr_right,
        curr_bottom,
        curr_top,
        x_scaler,
        y_scaler,
        curr_column,
        curr_row,
    ):
        # docs_tag: begin_interpolate_bboxes
        center_x = (curr_column * x_scaler + self.offset) / self.bbox_norm
        center_y = (curr_row * y_scaler + self.offset) / self.bbox_norm
        left = curr_left - center_x
        right = curr_right + center_x
        top = curr_top - center_y
        bottom = curr_bottom + center_y
        xmin = left * -self.bbox_norm
        xmax = right * self.bbox_norm
        ymin = top * -self.bbox_norm
        ymax = bottom * self.bbox_norm
        # docs_tag: end_interpolate_bboxes
        return [
            int(xmin.item()),
            int(ymin.item()),
            int(xmax.item() - xmin.item()),
            int(ymax.item() - ymin.item()),
        ]

    def __call__(self, raw_boxes, raw_scores, frame_nhwc):

        nvtx.push_range("postprocess.cvcuda")

        # docs_tag: begin_call_filterbboxcvcuda
        x_scaler = frame_nhwc.shape[2] / self.num_cols
        y_scaler = frame_nhwc.shape[1] / self.num_rows
        batch_size = raw_boxes.shape[0]
        filtered_bboxes = []
        # TODO Refactor and improve after adding NMS
        for b in range(batch_size):
            bboxes = []
            for c in range(self.num_classes):
                for y in range(self.num_rows):
                    for x in range(self.num_cols):
                        score = raw_scores[b][c][y][x]
                        if score > self.threshold:
                            bbox = self.interpolate_bboxes(
                                raw_boxes[b][c * 4][y][x],
                                raw_boxes[b][c * 4 + 2][y][x],
                                raw_boxes[b][c * 4 + 1][y][x],
                                raw_boxes[b][c * 4 + 3][y][x],
                                x_scaler,
                                y_scaler,
                                x,
                                y,
                            )
                            bboxes.append(bbox)
            filtered_bboxes.append(bboxes)
        # docs_tag: end_call_filterbboxcvcuda

        # Stage 5: render bounding boxes and Blur ROI's
        # docs_tag: start_outbuffer
        frame_nhwc = self.bboxutil(filtered_bboxes, frame_nhwc)
        if self.output_layout == "NCHW":
            render_output = cvcuda.reformat(frame_nhwc, "NCHW")
        else:
            assert self.output_layout == "NHWC"
            render_output = frame_nhwc

        if self.gpu_output:
            render_output = torch.as_tensor(
                render_output.cuda(), device="cuda:%d" % self.device_id
            )
        else:
            render_output = torch.as_tensor(render_output.cuda()).cpu().numpy()

        nvtx.pop_range()  # postprocess

        # Return 2 pieces of information:
        #   1. The original nhwc frame with bboxes rendered and ROI's blurred
        #   2. The bounding boxes predicted
        return (render_output, filtered_bboxes)
        # docs_tag: end_outbuffer


class BoundingBoxUtilsCvcuda:
    def __init__(self):
        # docs_tag: begin_init_cuosd_bboxes
        # Settings for the bounding boxes to be rendered
        self.border_color = (0, 255, 0, 255)
        self.fill_color = (0, 0, 255, 0)
        self.thickness = 5
        # kernel size for the blur ROI
        self.kernel_size = 7
        # docs_tag: end_init_cuosd_bboxes

    def __call__(self, bboxes, frame_nhwc):
        # docs_tag: begin_call_cuosd_bboxes
        batch_size = frame_nhwc.shape[0]
        num_boxes = []
        for b in range(batch_size):
            num_boxes.append(len(bboxes[b]))
        boxes = []
        blur_boxes = []
        # Create an array of bounding boxes with render settings.
        for b in range(batch_size):
            for i in range(num_boxes[b]):
                box = [
                    bboxes[b][i][0],
                    bboxes[b][i][1],
                    bboxes[b][i][2],
                    bboxes[b][i][3],
                ]
                boxes.append(
                    cvcuda.BndBoxI(
                        box=tuple(box),
                        thickness=self.thickness,
                        borderColor=self.border_color,
                        fillColor=self.fill_color,
                    )
                )
                blur_boxes.append(
                    cvcuda.BlurBoxI(box=tuple(box), kernelSize=self.kernel_size)
                )
            cusod_boxes = cvcuda.BndBoxesI(numBoxes=num_boxes, boxes=tuple(boxes))
            cuosd_blur_boxes = cvcuda.BlurBoxesI(
                numBoxes=num_boxes, boxes=tuple(blur_boxes)
            )

        # Render bounding boxes and blur the ROI inside the bounding box
        cvcuda.bndbox_into(frame_nhwc, frame_nhwc, cusod_boxes)

        # Invoke boxblur only when number of boxes > 0
        if cuosd_blur_boxes.numBoxes:
            cvcuda.boxblur_into(frame_nhwc, frame_nhwc, cuosd_blur_boxes)

        # docs_tag: end_call_cuosd_bboxes
        return frame_nhwc
