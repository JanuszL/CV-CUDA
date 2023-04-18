/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OsdElement.hpp"

#include <common/String.hpp>
#include <nvcv/OsdElement.h>
#include <pybind11/stl.h>

static std::ostream &operator<<(std::ostream &out, const NVCVBndBoxI &bbox)
{
    return out << "BndBox(x=" << bbox.rect.x << ",y=" << bbox.rect.y << ",width=" << bbox.rect.width << ",height=" << bbox.rect.height
            << ",thickness=" << bbox.thickness << ')';
}
static std::ostream &operator<<(std::ostream &out, const NVCVBndBoxesI &bboxes)
{
    for (int i = 0; i < bboxes.box_num; i++) {
        auto bbox = bboxes.boxes[i];
        out << "BndBoxes[" << i << "](x=" << bbox.rect.x << ",y=" << bbox.rect.y << ",width=" << bbox.rect.width << ",height=" << bbox.rect.height
            << ",thickness=" << bbox.thickness << ')';
    }
    return out;
}
static std::ostream &operator<<(std::ostream &out, const NVCVBlurBoxI &bbox)
{
    return out << "BlurBox(x=" << bbox.rect.x << ",y=" << bbox.rect.y << ",width=" << bbox.rect.width << ",height=" << bbox.rect.height
            << ",kernelSize=" << bbox.kernelSize << ')';
}
static std::ostream &operator<<(std::ostream &out, const NVCVBlurBoxesI &bboxes)
{
    for (int i = 0; i < bboxes.box_num; i++) {
        auto bbox = bboxes.boxes[i];
        out << "BlurBoxes[" << i << "](x=" << bbox.rect.x << ",y=" << bbox.rect.y << ",width=" << bbox.rect.width << ",height=" << bbox.rect.height
            << ",kernelSize=" << bbox.kernelSize << ')';
    }
    return out;
}
namespace nvcvpy::priv {

static NVCVRectI pytorect(py::tuple rect){
  if(rect.size() > 4 || rect.size() == 0) throw py::value_error("Invalid color size.");

  NVCVRectI ret;
  memset(&ret, 0, sizeof(ret));

  int* pr = (int*)&ret;
  for(size_t i = 0; i < rect.size(); ++i){
    pr[i] = rect[i].cast<int>();
  }
  return ret;
}

static NVCVColor pytocolor(py::tuple color){
  if(color.size() > 4 || color.size() == 0) throw py::value_error("Invalid color size.");

  NVCVColor ret;
  memset(&ret, 0, sizeof(ret));
  ret.a = 255;

  unsigned char* pr = (unsigned char*)&ret;
  for(size_t i = 0; i < color.size(); ++i){
    pr[i] = color[i].cast<unsigned char>();
  }
  return ret;
}

void ExportBndBox(py::module &m)
{
    using namespace py::literals;

    py::class_<NVCVBndBoxI>(m, "BndBoxI")
        .def(py::init([]() { return NVCVBndBoxI{}; }))
        .def(py::init(
                 [](py::tuple rect, int thickness, py::tuple borderColor, py::tuple fillColor)
                 {
                     NVCVBndBoxI bndbox;
                     bndbox.rect = pytorect(rect);
                     bndbox.borderColor = pytocolor(borderColor);
                     bndbox.fillColor = pytocolor(fillColor);
                     return bndbox;
                 }),
             "rect"_a, "thickness"_a, "borderColor"_a, "fillColor"_a)
        .def_readwrite("rect", &NVCVBndBoxI::rect)
        .def_readwrite("thickness", &NVCVBndBoxI::thickness)
        .def_readwrite("borderColor", &NVCVBndBoxI::borderColor)
        .def_readwrite("fillColor", &NVCVBndBoxI::fillColor)
        .def("__repr__", &util::ToString<NVCVBndBoxI>);

    py::class_<NVCVBndBoxesI>(m, "BndBoxesI")
        .def(py::init([]() { return NVCVBndBoxesI{}; }))
        .def(py::init(
                 [](std::vector<NVCVBndBoxI> bndboxes_vec)
                 {
                     NVCVBndBoxesI bndboxes;
                     bndboxes.box_num = bndboxes_vec.size();
                     bndboxes.boxes = bndboxes_vec.data();
                     return bndboxes;
                 }),
             "bndboxes"_a)
        .def_readwrite("box_num", &NVCVBndBoxesI::box_num)
        .def_readwrite("boxes", &NVCVBndBoxesI::boxes)
        .def("__repr__", &util::ToString<NVCVBndBoxesI>);
}

void ExportBoxBlur(py::module &m)
{
    using namespace py::literals;

    py::class_<NVCVBlurBoxI>(m, "BlurBoxI")
        .def(py::init([]() { return NVCVBlurBoxI{}; }))
        .def(py::init(
                 [](py::tuple rect, int kernelSize)
                 {
                     NVCVBlurBoxI blurbox;
                     blurbox.rect = pytorect(rect);
                     blurbox.kernelSize = kernelSize;
                     return blurbox;
                 }),
             "rect"_a, "kernelSize"_a)
        .def_readwrite("rect", &NVCVBlurBoxI::rect)
        .def_readwrite("kernelSize", &NVCVBlurBoxI::kernelSize)
        .def("__repr__", &util::ToString<NVCVBlurBoxI>);

    py::class_<NVCVBlurBoxesI>(m, "BlurBoxesI")
        .def(py::init([]() { return NVCVBlurBoxesI{}; }))
        .def(py::init(
                 [](std::vector<NVCVBlurBoxI> blurboxes_vec)
                 {
                     NVCVBlurBoxesI blurboxes;
                     blurboxes.box_num = blurboxes_vec.size();
                     blurboxes.boxes = new NVCVBlurBoxI[blurboxes.box_num];
                     memcpy(blurboxes.boxes, blurboxes_vec.data(), blurboxes_vec.size() * sizeof(NVCVBlurBoxI));
                     return blurboxes;
                 }),
             "blurboxes"_a)
        .def_readwrite("box_num", &NVCVBlurBoxesI::box_num)
        .def_readwrite("boxes", &NVCVBlurBoxesI::boxes)
        .def("__repr__", &util::ToString<NVCVBlurBoxesI>);
}

} // namespace nvcvpy::priv
