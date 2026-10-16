# Third-party notices

The MIT License in [`LICENSE`](LICENSE) covers only the original source code and
documentation written for BrandPose. It does not cover the third-party
libraries, model weights, videos or images listed below, which keep their own
licenses. This repository does not bundle any of them: libraries are installed
from `requirements.txt`, weights are downloaded at runtime and videos are
downloaded locally (see `data/videos/SOURCES.md`).

## Important: AGPL-3.0 and non-commercial components

The scripts import **Ultralytics YOLO** (`ultralytics` package and the
`yolo11m-pose.pt` / `yolo11m-seg.pt` weights), licensed under **AGPL-3.0**.
Distributing a program that combines BrandPose with Ultralytics, or offering it
as a network service, must comply with AGPL-3.0 (including releasing the
complete source of the combined work under AGPL-3.0), unless you obtain an
[Ultralytics Enterprise License](https://www.ultralytics.com/license) or replace
the detector. BrandPose's own code remains MIT and can be reused without
Ultralytics.

The default depth model in `scripts/promo/depth.py`,
**Depth Anything V2 Metric Indoor Base**, has weights licensed under
**CC-BY-NC-4.0** (non-commercial use only). For commercial use, switch to the
Small variant (Apache-2.0) or another model.

## Python libraries (`requirements.txt`)

| Package | License |
|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** |
| [torch](https://github.com/pytorch/pytorch) | BSD-3-Clause |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 |
| [transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| [opencv-python](https://github.com/opencv/opencv-python) | MIT (wrapper); OpenCV is Apache-2.0 |
| [numpy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [scipy](https://github.com/scipy/scipy) | BSD-3-Clause |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD-3-Clause |
| [pillow](https://github.com/python-pillow/Pillow) | MIT-CMU (HPND) |
| [lap](https://github.com/gatagat/lap) | BSD-2-Clause |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Unlicense |

## Models and weights (downloaded at runtime)

| Model | Source | License |
|---|---|---|
| YOLO11 pose and seg | Ultralytics | **AGPL-3.0** |
| Gaze-LLE | [fkryan/gazelle](https://github.com/fkryan/gazelle) (via `torch.hub`) | MIT |
| DINOv2 backbone (used by Gaze-LLE) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 |
| Depth Anything V2 Metric Indoor Base | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | **CC-BY-NC-4.0** |

## Images and videos

- The images in `docs/img/` and the demo video are derived from a public
  demonstration of the ACTi E43A camera
  ([YouTube](https://www.youtube.com/watch?v=-1bRhYjw1qE)). The footage is
  copyright ACTi Corporation and is **not** covered by the MIT License.
- Test videos from Mixkit, Pexels and the MERL Shopping dataset are not
  included in this repository and keep their own licenses. See
  [`data/videos/SOURCES.md`](data/videos/SOURCES.md).
- The fonts used to render the video (Avenir Next, Menlo, DIN Condensed) are
  macOS system fonts and are not distributed here.

License information was checked on 2026-10-16. Always confirm the current
license of each component before redistributing or using it commercially.
