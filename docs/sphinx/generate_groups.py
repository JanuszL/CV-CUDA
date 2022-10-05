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

from pathlib import Path
import xml.etree.ElementTree as ET
import os
import sys

outdir = Path(sys.argv[1])

if not os.path.exists(outdir):
  os.makedirs(outdir)

xmlRoot = sys.argv[2]

for i in os.listdir(xmlRoot):
    group_path = os.path.join(xmlRoot,i)
    if os.path.isfile(group_path) and 'group__' in i:
        tree = ET.parse(group_path)
        root = tree.getroot()
        for compounddef in root.iter('compounddef'):
            group_name = compounddef.attrib['id']
            group_label = compounddef.find('compoundname').text
            group_title = compounddef.find('title').text
            outfile = outdir / (group_name + ".rst")
            output = ":orphan:\n\n"
            output += group_title + "\n"
            output += "=" * len(group_title) + "\n\n"
            output += f".. doxygengroup:: {group_label}\n"
            output += f"   :project: cvcuda\n"
            outfile.write_text(output)
