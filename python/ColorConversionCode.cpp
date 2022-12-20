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

#include "ColorConversionCode.hpp"

#include <nvcv/operators/Types.h>

namespace nv::cvpy {

void ExportColorConversionCode(py::module &m)
{
    py::enum_<NVCVColorConversionCode>(m, "ColorConversion")
        .value("BGR2BGRA", NVCV_COLOR_BGR2BGRA)
        .value("RGB2RGBA", NVCV_COLOR_RGB2RGBA)
        .value("BGRA2BGR", NVCV_COLOR_BGRA2BGR)
        .value("RGBA2RGB", NVCV_COLOR_RGBA2RGB)
        .value("BGR2RGBA", NVCV_COLOR_BGR2RGBA)
        .value("RGB2BGRA", NVCV_COLOR_RGB2BGRA)
        .value("RGBA2BGR", NVCV_COLOR_RGBA2BGR)
        .value("BGRA2RGB", NVCV_COLOR_BGRA2RGB)
        .value("BGR2RGB", NVCV_COLOR_BGR2RGB)
        .value("RGB2BGR", NVCV_COLOR_RGB2BGR)
        .value("BGRA2RGBA", NVCV_COLOR_BGRA2RGBA)
        .value("RGBA2BGRA", NVCV_COLOR_RGBA2BGRA)
        .value("BGR2GRAY", NVCV_COLOR_BGR2GRAY)
        .value("RGB2GRAY", NVCV_COLOR_RGB2GRAY)
        .value("GRAY2BGR", NVCV_COLOR_GRAY2BGR)
        .value("GRAY2RGB", NVCV_COLOR_GRAY2RGB)
        .value("GRAY2BGRA", NVCV_COLOR_GRAY2BGRA)
        .value("GRAY2RGBA", NVCV_COLOR_GRAY2RGBA)
        .value("BGRA2GRAY", NVCV_COLOR_BGRA2GRAY)
        .value("RGBA2GRAY", NVCV_COLOR_RGBA2GRAY)
        .value("BGR2BGR565", NVCV_COLOR_BGR2BGR565)
        .value("RGB2BGR565", NVCV_COLOR_RGB2BGR565)
        .value("BGR5652BGR", NVCV_COLOR_BGR5652BGR)
        .value("BGR5652RGB", NVCV_COLOR_BGR5652RGB)
        .value("BGRA2BGR565", NVCV_COLOR_BGRA2BGR565)
        .value("RGBA2BGR565", NVCV_COLOR_RGBA2BGR565)
        .value("BGR5652BGRA", NVCV_COLOR_BGR5652BGRA)
        .value("BGR5652RGBA", NVCV_COLOR_BGR5652RGBA)
        .value("GRAY2BGR565", NVCV_COLOR_GRAY2BGR565)
        .value("BGR5652GRAY", NVCV_COLOR_BGR5652GRAY)
        .value("BGR2BGR555", NVCV_COLOR_BGR2BGR555)
        .value("RGB2BGR555", NVCV_COLOR_RGB2BGR555)
        .value("BGR5552BGR", NVCV_COLOR_BGR5552BGR)
        .value("BGR5552RGB", NVCV_COLOR_BGR5552RGB)
        .value("BGRA2BGR555", NVCV_COLOR_BGRA2BGR555)
        .value("RGBA2BGR555", NVCV_COLOR_RGBA2BGR555)
        .value("BGR5552BGRA", NVCV_COLOR_BGR5552BGRA)
        .value("BGR5552RGBA", NVCV_COLOR_BGR5552RGBA)
        .value("GRAY2BGR555", NVCV_COLOR_GRAY2BGR555)
        .value("BGR5552GRAY", NVCV_COLOR_BGR5552GRAY)
        .value("BGR2XYZ", NVCV_COLOR_BGR2XYZ)
        .value("RGB2XYZ", NVCV_COLOR_RGB2XYZ)
        .value("XYZ2BGR", NVCV_COLOR_XYZ2BGR)
        .value("XYZ2RGB", NVCV_COLOR_XYZ2RGB)
        .value("BGR2YCrCb", NVCV_COLOR_BGR2YCrCb)
        .value("RGB2YCrCb", NVCV_COLOR_RGB2YCrCb)
        .value("YCrCb2BGR", NVCV_COLOR_YCrCb2BGR)
        .value("YCrCb2RGB", NVCV_COLOR_YCrCb2RGB)
        .value("BGR2HSV", NVCV_COLOR_BGR2HSV)
        .value("RGB2HSV", NVCV_COLOR_RGB2HSV)
        .value("BGR2Lab", NVCV_COLOR_BGR2Lab)
        .value("RGB2Lab", NVCV_COLOR_RGB2Lab)
        .value("BGR2Luv", NVCV_COLOR_BGR2Luv)
        .value("RGB2Luv", NVCV_COLOR_RGB2Luv)
        .value("BGR2HLS", NVCV_COLOR_BGR2HLS)
        .value("RGB2HLS", NVCV_COLOR_RGB2HLS)
        .value("HSV2BGR", NVCV_COLOR_HSV2BGR)
        .value("HSV2RGB", NVCV_COLOR_HSV2RGB)
        .value("Lab2BGR", NVCV_COLOR_Lab2BGR)
        .value("Lab2RGB", NVCV_COLOR_Lab2RGB)
        .value("Luv2BGR", NVCV_COLOR_Luv2BGR)
        .value("Luv2RGB", NVCV_COLOR_Luv2RGB)
        .value("HLS2BGR", NVCV_COLOR_HLS2BGR)
        .value("HLS2RGB", NVCV_COLOR_HLS2RGB)
        .value("BGR2HSV_FULL", NVCV_COLOR_BGR2HSV_FULL)
        .value("RGB2HSV_FULL", NVCV_COLOR_RGB2HSV_FULL)
        .value("BGR2HLS_FULL", NVCV_COLOR_BGR2HLS_FULL)
        .value("RGB2HLS_FULL", NVCV_COLOR_RGB2HLS_FULL)
        .value("HSV2BGR_FULL", NVCV_COLOR_HSV2BGR_FULL)
        .value("HSV2RGB_FULL", NVCV_COLOR_HSV2RGB_FULL)
        .value("HLS2BGR_FULL", NVCV_COLOR_HLS2BGR_FULL)
        .value("HLS2RGB_FULL", NVCV_COLOR_HLS2RGB_FULL)
        .value("LBGR2Lab", NVCV_COLOR_LBGR2Lab)
        .value("LRGB2Lab", NVCV_COLOR_LRGB2Lab)
        .value("LBGR2Luv", NVCV_COLOR_LBGR2Luv)
        .value("LRGB2Luv", NVCV_COLOR_LRGB2Luv)
        .value("Lab2LBGR", NVCV_COLOR_Lab2LBGR)
        .value("Lab2LRGB", NVCV_COLOR_Lab2LRGB)
        .value("Luv2LBGR", NVCV_COLOR_Luv2LBGR)
        .value("Luv2LRGB", NVCV_COLOR_Luv2LRGB)
        .value("BGR2YUV", NVCV_COLOR_BGR2YUV)
        .value("RGB2YUV", NVCV_COLOR_RGB2YUV)
        .value("YUV2BGR", NVCV_COLOR_YUV2BGR)
        .value("YUV2RGB", NVCV_COLOR_YUV2RGB)
        .value("YUV2RGB_NV12", NVCV_COLOR_YUV2RGB_NV12)
        .value("YUV2BGR_NV12", NVCV_COLOR_YUV2BGR_NV12)
        .value("YUV2RGB_NV21", NVCV_COLOR_YUV2RGB_NV21)
        .value("YUV2BGR_NV21", NVCV_COLOR_YUV2BGR_NV21)
        .value("YUV420sp2RGB", NVCV_COLOR_YUV420sp2RGB)
        .value("YUV420sp2BGR", NVCV_COLOR_YUV420sp2BGR)
        .value("YUV2RGBA_NV12", NVCV_COLOR_YUV2RGBA_NV12)
        .value("YUV2BGRA_NV12", NVCV_COLOR_YUV2BGRA_NV12)
        .value("YUV2RGBA_NV21", NVCV_COLOR_YUV2RGBA_NV21)
        .value("YUV2BGRA_NV21", NVCV_COLOR_YUV2BGRA_NV21)
        .value("YUV420sp2RGBA", NVCV_COLOR_YUV420sp2RGBA)
        .value("YUV420sp2BGRA", NVCV_COLOR_YUV420sp2BGRA)
        .value("YUV2RGB_YV12", NVCV_COLOR_YUV2RGB_YV12)
        .value("YUV2BGR_YV12", NVCV_COLOR_YUV2BGR_YV12)
        .value("YUV2RGB_IYUV", NVCV_COLOR_YUV2RGB_IYUV)
        .value("YUV2BGR_IYUV", NVCV_COLOR_YUV2BGR_IYUV)
        .value("YUV2RGB_I420", NVCV_COLOR_YUV2RGB_I420)
        .value("YUV2BGR_I420", NVCV_COLOR_YUV2BGR_I420)
        .value("YUV420p2RGB", NVCV_COLOR_YUV420p2RGB)
        .value("YUV420p2BGR", NVCV_COLOR_YUV420p2BGR)
        .value("YUV2RGBA_YV12", NVCV_COLOR_YUV2RGBA_YV12)
        .value("YUV2BGRA_YV12", NVCV_COLOR_YUV2BGRA_YV12)
        .value("YUV2RGBA_IYUV", NVCV_COLOR_YUV2RGBA_IYUV)
        .value("YUV2BGRA_IYUV", NVCV_COLOR_YUV2BGRA_IYUV)
        .value("YUV2RGBA_I420", NVCV_COLOR_YUV2RGBA_I420)
        .value("YUV2BGRA_I420", NVCV_COLOR_YUV2BGRA_I420)
        .value("YUV420p2RGBA", NVCV_COLOR_YUV420p2RGBA)
        .value("YUV420p2BGRA", NVCV_COLOR_YUV420p2BGRA)
        .value("YUV2GRAY_420", NVCV_COLOR_YUV2GRAY_420)
        .value("YUV2GRAY_NV21", NVCV_COLOR_YUV2GRAY_NV21)
        .value("YUV2GRAY_NV12", NVCV_COLOR_YUV2GRAY_NV12)
        .value("YUV2GRAY_YV12", NVCV_COLOR_YUV2GRAY_YV12)
        .value("YUV2GRAY_IYUV", NVCV_COLOR_YUV2GRAY_IYUV)
        .value("YUV2GRAY_I420", NVCV_COLOR_YUV2GRAY_I420)
        .value("YUV420sp2GRAY", NVCV_COLOR_YUV420sp2GRAY)
        .value("YUV420p2GRAY", NVCV_COLOR_YUV420p2GRAY)
        .value("YUV2RGB_UYVY", NVCV_COLOR_YUV2RGB_UYVY)
        .value("YUV2BGR_UYVY", NVCV_COLOR_YUV2BGR_UYVY)
        .value("YUV2RGB_Y422", NVCV_COLOR_YUV2RGB_Y422)
        .value("YUV2BGR_Y422", NVCV_COLOR_YUV2BGR_Y422)
        .value("YUV2RGB_UYNV", NVCV_COLOR_YUV2RGB_UYNV)
        .value("YUV2BGR_UYNV", NVCV_COLOR_YUV2BGR_UYNV)
        .value("YUV2RGBA_UYVY", NVCV_COLOR_YUV2RGBA_UYVY)
        .value("YUV2BGRA_UYVY", NVCV_COLOR_YUV2BGRA_UYVY)
        .value("YUV2RGBA_Y422", NVCV_COLOR_YUV2RGBA_Y422)
        .value("YUV2BGRA_Y422", NVCV_COLOR_YUV2BGRA_Y422)
        .value("YUV2RGBA_UYNV", NVCV_COLOR_YUV2RGBA_UYNV)
        .value("YUV2BGRA_UYNV", NVCV_COLOR_YUV2BGRA_UYNV)
        .value("YUV2RGB_YUY2", NVCV_COLOR_YUV2RGB_YUY2)
        .value("YUV2BGR_YUY2", NVCV_COLOR_YUV2BGR_YUY2)
        .value("YUV2RGB_YVYU", NVCV_COLOR_YUV2RGB_YVYU)
        .value("YUV2BGR_YVYU", NVCV_COLOR_YUV2BGR_YVYU)
        .value("YUV2RGB_YUYV", NVCV_COLOR_YUV2RGB_YUYV)
        .value("YUV2BGR_YUYV", NVCV_COLOR_YUV2BGR_YUYV)
        .value("YUV2RGB_YUNV", NVCV_COLOR_YUV2RGB_YUNV)
        .value("YUV2BGR_YUNV", NVCV_COLOR_YUV2BGR_YUNV)
        .value("YUV2RGBA_YUY2", NVCV_COLOR_YUV2RGBA_YUY2)
        .value("YUV2BGRA_YUY2", NVCV_COLOR_YUV2BGRA_YUY2)
        .value("YUV2RGBA_YVYU", NVCV_COLOR_YUV2RGBA_YVYU)
        .value("YUV2BGRA_YVYU", NVCV_COLOR_YUV2BGRA_YVYU)
        .value("YUV2RGBA_YUYV", NVCV_COLOR_YUV2RGBA_YUYV)
        .value("YUV2BGRA_YUYV", NVCV_COLOR_YUV2BGRA_YUYV)
        .value("YUV2RGBA_YUNV", NVCV_COLOR_YUV2RGBA_YUNV)
        .value("YUV2BGRA_YUNV", NVCV_COLOR_YUV2BGRA_YUNV)
        .value("YUV2GRAY_UYVY", NVCV_COLOR_YUV2GRAY_UYVY)
        .value("YUV2GRAY_YUY2", NVCV_COLOR_YUV2GRAY_YUY2)
        .value("YUV2GRAY_Y422", NVCV_COLOR_YUV2GRAY_Y422)
        .value("YUV2GRAY_UYNV", NVCV_COLOR_YUV2GRAY_UYNV)
        .value("YUV2GRAY_YVYU", NVCV_COLOR_YUV2GRAY_YVYU)
        .value("YUV2GRAY_YUYV", NVCV_COLOR_YUV2GRAY_YUYV)
        .value("YUV2GRAY_YUNV", NVCV_COLOR_YUV2GRAY_YUNV)
        .value("RGBA2mRGBA", NVCV_COLOR_RGBA2mRGBA)
        .value("mRGBA2RGBA", NVCV_COLOR_mRGBA2RGBA)
        .value("RGB2YUV_I420", NVCV_COLOR_RGB2YUV_I420)
        .value("BGR2YUV_I420", NVCV_COLOR_BGR2YUV_I420)
        .value("RGB2YUV_IYUV", NVCV_COLOR_RGB2YUV_IYUV)
        .value("BGR2YUV_IYUV", NVCV_COLOR_BGR2YUV_IYUV)
        .value("RGBA2YUV_I420", NVCV_COLOR_RGBA2YUV_I420)
        .value("BGRA2YUV_I420", NVCV_COLOR_BGRA2YUV_I420)
        .value("RGBA2YUV_IYUV", NVCV_COLOR_RGBA2YUV_IYUV)
        .value("BGRA2YUV_IYUV", NVCV_COLOR_BGRA2YUV_IYUV)
        .value("RGB2YUV_YV12", NVCV_COLOR_RGB2YUV_YV12)
        .value("BGR2YUV_YV12", NVCV_COLOR_BGR2YUV_YV12)
        .value("RGBA2YUV_YV12", NVCV_COLOR_RGBA2YUV_YV12)
        .value("BGRA2YUV_YV12", NVCV_COLOR_BGRA2YUV_YV12)
        .value("BayerBG2BGR", NVCV_COLOR_BayerBG2BGR)
        .value("BayerGB2BGR", NVCV_COLOR_BayerGB2BGR)
        .value("BayerRG2BGR", NVCV_COLOR_BayerRG2BGR)
        .value("BayerGR2BGR", NVCV_COLOR_BayerGR2BGR)
        .value("BayerBG2RGB", NVCV_COLOR_BayerBG2RGB)
        .value("BayerGB2RGB", NVCV_COLOR_BayerGB2RGB)
        .value("BayerRG2RGB", NVCV_COLOR_BayerRG2RGB)
        .value("BayerGR2RGB", NVCV_COLOR_BayerGR2RGB)
        .value("BayerBG2GRAY", NVCV_COLOR_BayerBG2GRAY)
        .value("BayerGB2GRAY", NVCV_COLOR_BayerGB2GRAY)
        .value("BayerRG2GRAY", NVCV_COLOR_BayerRG2GRAY)
        .value("BayerGR2GRAY", NVCV_COLOR_BayerGR2GRAY)
        .value("BayerBG2BGR_VNG", NVCV_COLOR_BayerBG2BGR_VNG)
        .value("BayerGB2BGR_VNG", NVCV_COLOR_BayerGB2BGR_VNG)
        .value("BayerRG2BGR_VNG", NVCV_COLOR_BayerRG2BGR_VNG)
        .value("BayerGR2BGR_VNG", NVCV_COLOR_BayerGR2BGR_VNG)
        .value("BayerBG2RGB_VNG", NVCV_COLOR_BayerBG2RGB_VNG)
        .value("BayerGB2RGB_VNG", NVCV_COLOR_BayerGB2RGB_VNG)
        .value("BayerRG2RGB_VNG", NVCV_COLOR_BayerRG2RGB_VNG)
        .value("BayerGR2RGB_VNG", NVCV_COLOR_BayerGR2RGB_VNG)
        .value("BayerBG2BGR_EA", NVCV_COLOR_BayerBG2BGR_EA)
        .value("BayerGB2BGR_EA", NVCV_COLOR_BayerGB2BGR_EA)
        .value("BayerRG2BGR_EA", NVCV_COLOR_BayerRG2BGR_EA)
        .value("BayerGR2BGR_EA", NVCV_COLOR_BayerGR2BGR_EA)
        .value("BayerBG2RGB_EA", NVCV_COLOR_BayerBG2RGB_EA)
        .value("BayerGB2RGB_EA", NVCV_COLOR_BayerGB2RGB_EA)
        .value("BayerRG2RGB_EA", NVCV_COLOR_BayerRG2RGB_EA)
        .value("BayerGR2RGB_EA", NVCV_COLOR_BayerGR2RGB_EA)
        .value("COLORCVT_MAX", NVCV_COLOR_COLORCVT_MAX)
        .value("RGB2YUV_NV12", NVCV_COLOR_RGB2YUV_NV12)
        .value("BGR2YUV_NV12", NVCV_COLOR_BGR2YUV_NV12)
        .value("RGB2YUV_NV21", NVCV_COLOR_RGB2YUV_NV21)
        .value("RGB2YUV420sp", NVCV_COLOR_RGB2YUV420sp)
        .value("BGR2YUV_NV21", NVCV_COLOR_BGR2YUV_NV21)
        .value("BGR2YUV420sp", NVCV_COLOR_BGR2YUV420sp)
        .value("RGBA2YUV_NV12", NVCV_COLOR_RGBA2YUV_NV12)
        .value("BGRA2YUV_NV12", NVCV_COLOR_BGRA2YUV_NV12)
        .value("RGBA2YUV_NV21", NVCV_COLOR_RGBA2YUV_NV21)
        .value("RGBA2YUV420sp", NVCV_COLOR_RGBA2YUV420sp)
        .value("BGRA2YUV_NV21", NVCV_COLOR_BGRA2YUV_NV21)
        .value("BGRA2YUV420sp", NVCV_COLOR_BGRA2YUV420sp)
        .value("CVT_MAX", NVCV_COLORCVT_MAX);
}

} // namespace nv::cvpy
