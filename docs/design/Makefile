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

.PHONY: all

DIAGRAMS = allocator_classes \
	   resource_allocator_classes \
	   image_classes \
	   image_data_classes

all: $(addsuffix .eps,$(DIAGRAMS))

%.eps: %.puml
	plantuml -teps $^ $@

clean:
	rm -f $(addsuffix .eps,$(DIAGRAMS))
