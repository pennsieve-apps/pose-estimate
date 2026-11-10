# Third-party software and model weights

This app's own code is Apache-2.0 (see `LICENSE`). The image it builds contains the components
below, installed from PyPI at the pinned versions in `requirements*.txt`, and four model weight
files from Hugging Face; their licences apply to the image.

## Model weights: academic, non-commercial use only

**SuperAnimal weights: non-commercial use.** The image bakes in four files of the Mathis lab's
SuperAnimal models (Ye et al. 2024, *Nat Commun* 15:5165, doi:10.1038/s41467-024-48792-2):

| File | Model | Source (pinned revision) | SHA-256 |
|---|---|---|---|
| `superanimal_topviewmouse_hrnet_w32.pt` | SuperAnimal-TopViewMouse, HRNet-w32 pose model (27 keypoints) | [mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse) @ `d443257e4b1145f7439e0953c2015d60fcdc1185` | `5e7ed2b4e568fa93e2b1404251f9b060f9bd3ff5d68712f57c8ddb766d0cf6ed` |
| `superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2.pt` | SuperAnimal-TopViewMouse, Faster R-CNN ResNet-50 FPN v2 detector | same | `1175e31a5486731734948da55e4b25adcee84da4e36eafc5768a771809d536e5` |
| `superanimal_quadruped_hrnet_w32.pt` | SuperAnimal-Quadruped, HRNet-w32 pose model (39 keypoints) | [mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped) @ `1ad30fb80cd666f1e5c91578d1cf63bccfa84f80` | `a42e58d56bb32f8b4f4fe17ea9ed9511cebc7b7949ac56b97fa6f3a49587c31e` |
| `superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt` | SuperAnimal-Quadruped, Faster R-CNN ResNet-50 FPN v2 detector | same | `8c723bc3c357f9bfe87d3a23c30218be5d450d2289fd1e33f115327cf6c53d54` |

Copyright 2023 by Mackenzie Mathis, Shaokai Ye, and contributors. Licence: **"Modified MIT"**, a
fully-paid, non-exclusive, non-transferable licence **for academic, non-commercial purposes only**;
the weights may not be used to harm any animal deliberately. The full text, as published on the
model cards, is in [`licenses/SuperAnimal-LICENSE.txt`](licenses/SuperAnimal-LICENSE.txt) and in the
image at `/processor/licenses/`. For a commercial licence contact Prof. Mackenzie W. Mathis or the
EPFL technology transfer office (tto@epfl.ch). Some of the training images are themselves under
non-commercial terms (see the model cards).

Because the licence is non-transferable, the coordinator decided (builder brief, "Licences") that
the image keeps the licence text, the app says "SuperAnimal weights: non-commercial use" in its
description and README, and the image is distributed for academic use only. **The licence-clean
path** is the `model` input: with the lab's own DeepLabCut model the SuperAnimal weights are not
used (they are still present in the image).

## Software

| Component | What | Authors / copyright | Licence | Source |
|---|---|---|---|---|
| DeepLabCut 3.0.2 | Pose estimation engine (PyTorch): SuperAnimal configs, top-down inference runners, video adaptation | A. & M.W. Mathis Labs and contributors (M-Lab of Adaptive Intelligence; Mathis Group for Computational Neuroscience and AI, EPFL) | LGPL-3.0-or-later ([`licenses/DeepLabCut-LGPL-3.0.txt`](licenses/DeepLabCut-LGPL-3.0.txt), plus the GPL-3.0 text it incorporates, [`licenses/GPL-3.0.txt`](licenses/GPL-3.0.txt)); used unmodified as a library | https://github.com/DeepLabCut/DeepLabCut |
| dlclibrary 0.0.12 | DeepLabCut model zoo index | DeepLabCut contributors | LGPL-3.0 | https://github.com/DeepLabCut/DLClibrary |
| PyTorch 2.14.1, torchvision 0.29.1 | Neural network runtime; Faster R-CNN implementation | PyTorch contributors (Meta and others) | BSD-3-Clause (torch bundles further permissive licences) | https://pytorch.org |
| NVIDIA CUDA runtime wheels (gpu variant only: cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NCCL, NVRTC, nvJitLink, NVTX, cuFile, NVSHMEM, cuSPARSELt), triton 3.8.0 | GPU libraries pulled in by the CUDA build of torch | NVIDIA; OpenAI (triton) | NVIDIA proprietary redistributable (CUDA EULA); triton MIT | https://pypi.org/project/nvidia-cudnn-cu12 |
| timm 1.0.31 | HRNet backbone | Ross Wightman and contributors | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |
| albumentations 1.4.3 | Image transforms used by DeepLabCut | Buslaev et al. | MIT | https://github.com/albumentations-team/albumentations |
| pynwb 4.2.0, hdmf 6.2.0 | Writing NWB | Tritt, Ly, Ruebel, Dichter et al. (LBNL) | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb |
| ndx-pose 0.4.0 | NWB pose extension (PoseEstimation, Skeleton) | Ryan Ly, Ben Dichter, Alexander Mathis, Liezl Maree, Chris Brozdowski, Heberto Mayorquin, Talmo Pereira, Elizabeth Berrigan, Paul Adkisson, Alessandra Trapani | BSD-3-Clause | https://github.com/rly/ndx-pose |
| opencv-python-headless 4.14.0.94 | Video decoding and probing, drawing, and the QC clip (VP9 in WebM) | OpenCV team; FFmpeg developers; WebM project (libvpx); VideoLAN (dav1d); AOM (libaom) | Apache-2.0; the wheel bundles FFmpeg (libavcodec 62.28.102 and friends) configured with `--enable-libvpx --enable-libdav1d --enable-openssl` and **without** `--enable-gpl` (its license string: "LGPL version 2.1 or later"), libvpx (BSD-3-Clause), dav1d (BSD-2-Clause), libaom (BSD-2-Clause + AOM patent licence), OpenSSL 1.1.1k (OpenSSL licence), OpenBLAS (BSD-3-Clause) and libgfortran (GPL-3.0 with the GCC Runtime Library Exception, which permits this use); details in the wheel's `LICENSE-3RD-PARTY.txt` | https://github.com/opencv/opencv-python |
| huggingface_hub 2.4.0 | Build-time download of the weights (offline at run time) | Hugging Face | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| NumPy 2.5.4, SciPy 1.18.1, pandas 2.3.3, h5py 3.16.0, PyTables 3.11.1, numexpr, Bottleneck, blosc2 | Arrays, tables, HDF5 (the DeepLabCut .h5) | their developers | BSD-3-Clause (h5py, PyTables, blosc2: BSD) | PyPI |
| Matplotlib 3.11.2 | Figures | Matplotlib development team | Matplotlib licence (PSF-based) | https://matplotlib.org |
| scikit-image, scikit-learn, statsmodels, numba, llvmlite, pycocotools, filterpy, einops, networkx, pillow, pydantic, ruamel.yaml, PyYAML, tqdm, rich, safetensors, imageio, tifffile, ... | Dependencies of DeepLabCut (versions in `requirements.txt`) | their developers | BSD / MIT / Apache-2.0 / HPND (Pillow); tqdm MPL-2.0 AND MIT; certifi MPL-2.0 | PyPI |
| python:3.12.16-slim-bookworm | Base image (Debian 12 + CPython 3.12) | PSF; Debian | PSF; Debian packages under their own free licences | https://hub.docker.com/_/python |

**Nothing GPL in the app.** Video is decoded and the QC clip encoded only through OpenCV's LGPL
FFmpeg. Left out on purpose: PyAV (its wheels link an FFmpeg built with libx264, GPL, into the
process) and imageio-ffmpeg (a declared dependency of DeepLabCut that DeepLabCut's Python code does
not import; its `ffmpeg` executable is a GPL-3.0 build). DeepLabCut's own video utilities that call
an `ffmpeg` program (labelled videos, cropping) are not used, and there is no such program in the
image.
`licenses/GPL-3.0.txt` is there because the LGPL-3.0 of DeepLabCut and dlclibrary incorporates the
GPL-3.0's terms and requires its text. As in any Debian-based image, the base system's own
command-line tools (bash, coreutils, ...) are GPL programs; the app does not link them.

## Not bundled

- **Demo data** (not in the image): DANDI:002047 version 0.261106.0903, "Hippocampal CA1 one-photon
  calcium imaging and pose-tracked behavior in wild-type and 5xFAD mice across four behavioral
  paradigms" (Plusnin, Pospelov; contributors Rogozhnikova, Sotskov, Ivanova, Savelev, Toropova,
  Ivashkina, Anokhin), doi:10.48324/dandi.002047/0.261106.0903, CC-BY-4.0. A 2-second excerpt of one of
  its behaviour videos (session sub-F01 ses-OF-F01-S1, frames 2998-3057) and the authors' pose for
  those frames are in the repository's `tests/data/` (CC-BY-4.0, attribution as above); they are test
  fixtures, not part of the image.
