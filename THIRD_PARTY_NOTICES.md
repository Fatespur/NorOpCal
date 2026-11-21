# Third-Party Notices and Attribution

The NorOpCal repository contains **in-house reimplementations** of the three
anomaly detectors used in the paper, plus adapters that load **pretrained
third-party backbone weights**. No third-party source files are vendored into
this repository; the notices below cover the external components referenced or
downloaded at runtime.

## EfficientAD

- Location: `src/shiftad/models/efficientad/` (in-house implementation).
- The module docstring references the reference implementation at
  <https://github.com/nelson1425/EfficientAD> (MIT License) as prior art.
- Paper: Batzner, Heckler, Konig, "EfficientAD: Accurate Visual Anomaly
  Detection at Millisecond-Level Latencies", WACV 2024.
- The paper's EfficientAD instance is trained IN-HOUSE (70,000 steps,
  source-normal-only); no official pretrained weights are used or implied.

## DINOv2 backbone

- Location: `src/shiftad/backbones/dinov2.py` loads the OFFICIAL model via
  `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` when weight
  download/cache use is explicitly permitted by configuration.
- DINOv2 code is released by Meta AI Research under the Apache License 2.0;
  pretrained checkpoint terms follow the official repository. Users must
  comply with those licenses when enabling downloads.
- Paper: Oquab et al., "DINOv2: Learning Robust Visual Features without
  Supervision", TMLR 2023.

## ResNet / Wide ResNet backbones (PatchCore)

- Location: `src/shiftad/backbones/resnet.py` uses torchvision ResNet-family
  models (BSD-3-Clause, Copyright (c) Soumith Chintala 2016 and torchvision
  contributors) and optionally their ImageNet-pretrained weights.
- PatchCore is an in-house implementation of the algorithm of Roth et al.,
  "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022); the
  original reference implementation is Apache-2.0. No source code was copied.

## Runtime libraries

torch (BSD-3), torchvision (BSD-3), numpy (BSD-3), pandas (BSD-3),
scipy (BSD-3), scikit-learn (BSD-3), matplotlib (PSF-based), PyYAML (MIT),
Pillow (HPND/MIT-CMU). See the respective distributions for full texts.

## Statement

Scientific pipeline code under `src/shiftad/`, `tools/`, and `scripts/` was
written for this project unless a notice above states otherwise.
