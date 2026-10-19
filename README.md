# BrandPose: where shoppers look, from an ordinary security camera

🌐 **English** · [Português (Brasil)](README.pt-BR.md)

> Personal project by **Claudio Escudero** ([linkedin.com/in/claudioescudero](https://www.linkedin.com/in/claudioescudero/)). Not a product.

The question behind the project: **can you estimate where each shopper looks in a store using only an ordinary security camera and open source AI models?**

The starting point was [this video from CactusXR (Auki)](https://x.com/CactusXR/status/2103355813444321655). In it, store cameras produce an attention heatmap over the shelves. This repository tries to reproduce the idea from scratch and documents the path, including what worked and what went wrong.

> 📹 **Video used in the demo:** [public demo of the ACTi E43A security camera in a US convenience store](https://www.youtube.com/watch?v=-1bRhYjw1qE) (YouTube, 60 s, 1080p, fixed camera). All processing below was done on this video. Footage © ACTi Corporation.

## Result

🎬 **Demo video (42 s):** [watch on YouTube](https://youtu.be/w3M8PlVcndU) · [download from the release](https://github.com/escudero/brandpose/releases/latest)

[![Watch the demo video on YouTube](docs/img/ao_vivo_play_en.jpg)](https://youtu.be/w3M8PlVcndU)

| Gaze estimated frame by frame | The scene in 3D (from a single image) | Attention report |
|---|---|---|
| ![Gaze beams and heat on the shelves](docs/img/ao_vivo_en.jpg) | ![3D point cloud of the store](docs/img/loja_3d_en.jpg) | ![Attention ranking by zone](docs/img/relatorio_en.jpg) |

Over 60 s from a real security camera in a convenience store:

| Zone | Attention | % | People |
|---|---|---|---|
| Candy and chocolate | 16.1 s | 32% | 8 |
| Sunglasses and accessories | 15.8 s | 31% | 10 |
| Back-wall snacks | 7.2 s | 14% | 7 |
| Coffee and juice | 2.8 s | 5% | 6 |
| Endcap | 2.5 s | 5% | 2 |

That is 797 estimated gazes, 17 shoppers and 9 zones with attention. These numbers are **model estimates**, with no manual ground truth (see the limitations).

## How it works

```mermaid
flowchart LR
    V["Fixed-camera video"] --> P["YOLO11-pose<br/>people + skeleton + ID"]
    V --> S["YOLO11-seg<br/>silhouettes"]
    P --> G["Gaze-LLE ViT-L<br/>per-person crop<br/>where they look"]
    G --> R["Attention rules<br/>edge · people · staff · zones"]
    S --> R
    V --> D["Depth Anything V2<br/>depth in meters"]
    R --> X["Render: heat, ranking,<br/>3D and synthesized soundtrack"]
    D --> X
    X --> MP4["Video + report"]
```

1. **People:** YOLO11-pose finds each person, the 17 body keypoints and an ID that follows the person frame by frame (ByteTrack, on every frame).
2. **Gaze:** [Gaze-LLE](https://github.com/fkryan/gazelle) (CVPR 2025) takes a **square crop around each head** and returns a heatmap of where the person is looking, 10 times per second.
3. **Rules:** the gaze point is smoothed over time and assigned to a store zone. Gazes that are not shelf attention are discarded (see below).
4. **3D:** Depth Anything V2 estimates the depth of one frame, which becomes a point cloud of about 500 thousand points.
5. **Video:** OpenCV, NumPy and Pillow draw each frame, FFmpeg encodes it, and the soundtrack is synthesized in code.

Details of each stage, with libraries, versions, formats and timings, are in [`docs/arquitetura.md`](docs/arquitetura.md) (in Portuguese).

## What I learned along the way

| Finding | Evidence | What changed |
|---|---|---|
| Squeezing the whole frame into 448×448 blinds the model | A 58 px head reached the model at about 13×23 px | Per-person crop: head 3.5 times larger and undistorted |
| The tracker lost people when running at 3 frames per second | At t=24 s, 4 people were detected but only 2 tracked | Tracking on every frame |
| **Gaze leaving the image sticks to the edge** | 34% of peaks were less than 24 px from the edge | A peak less than 32 px from the edge counts as "out of frame" |
| **Gaze landing on other people became "shelf attention"** | 25% of gazes fell on someone's silhouette | A target on a person counts toward no zone |
| The same person got new IDs | Tracks restarting up to 1.5 s later, less than 150 px away | Fragments merged: 20 tracks became 17 shoppers |
| The camera lens is unknown | Minimizing the floor residual always picked the narrowest field of view | Field of view where the shelves stand at 90° to the floor: 60° |
| The metric depth shrank the store | In the 3D from depth, the camera stood at 2.04 m and the shelves were 0.7 to 1.0 m tall | Camera calibrated from marked corners, with people as the ruler: 2.68 m high, 57° field of view and barrel distortion |

Without the edge and people rules, the ranking showed "Sunglasses" with 77 s and 36%, a false result.

## Through a shopper's eyes: the store in 3D

With the same data (people tracked frame by frame and their estimated gaze), I rebuilt the store in Blender and rendered what one shopper saw: the man in the blue jacket who walks in at 17 s.

🎬 **The shopper's view (18 s):** [watch on YouTube](https://youtu.be/iE7DxHnahjM)

[![Watch the shopper's view on YouTube](docs/img/pov_play_en.jpg)](https://youtu.be/iE7DxHnahjM)

| Real camera and 3D model, same calibrated view | The store modeled in Blender | His view, frame by frame |
|---|---|---|
| ![Security camera image next to the Blender render from the same viewpoint](docs/img/blender_cctv_en.jpg) | ![The store modeled in Blender, with a sign per zone](docs/img/blender_loja_en.jpg) | ![The shopper's view inside the 3D store](docs/img/blender_pov_en.jpg) |

1. **Calibration:** I marked the corners of each fixture on the empty-store image. The camera (height, angles, focal length and lens distortion) is fitted together with the position and size of each fixture, by least squares. People are the ruler: the nose sits about 1.49 m above the ankles.
2. **Modeling:** each fixture becomes a Blender model (gondolas, coolers, racks, counter) where the calibration placed it, with a sign showing the zone and its share of attention. Products, walls and lights are generic.
3. **Path and gaze:** each person's feet come from the skeleton, through the calibrated camera. The gaze becomes a ray from the security camera through the Gaze-LLE peak, and the point where it hits the 3D store is where the person looks.
4. **His camera:** eyes at 1.62 m, pointing where the smoothed gaze points (where he walks when there is no gaze), rendered in Cycles: 532 frames at 1280×720, about 8 s per frame on an M2 Pro.

The scripts are in [`scripts/blender/`](scripts/blender/). Method, numbers and limitations are in [`docs/experiments/2026-10-18-blender-pov.md`](docs/experiments/2026-10-18-blender-pov.md) (in Portuguese).

## Reproduce

Requirements: Python 3.13, FFmpeg, about 10 GB free. Tested on a Mac M2 Pro with 16 GB (MPS acceleration).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# source video (public demo of the ACTi E43A camera)
.venv/bin/yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]" \
    -o "data/videos/youtube/yt_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=-1bRhYjw1qE"

# simple simulation (recommended settings)
cd scripts
../.venv/bin/python gaze_sim_v2.py ../data/videos/youtube/yt_-1bRhYjw1qE.mp4 ../outputs/v2 \
    --crop --model gazelle_dinov2_vitl14_inout --smooth 0.5 --suppress-self --track-every-frame --max-seconds 60
```

The step by step for the full video (extraction, zones, depth, render and soundtrack) is in [`docs/experiments/2026-10-14-promo-v2.md`](docs/experiments/2026-10-14-promo-v2.md) (in Portuguese). The zones drawn for this camera are in [`config/zones_acti.json`](config/zones_acti.json).

The 3D store and the shopper's view start from that same extraction and also need [Blender](https://www.blender.org) 5.2. The commands are in [`docs/experiments/2026-10-18-blender-pov.md`](docs/experiments/2026-10-18-blender-pov.md#rodar) (in Portuguese).

## Structure

```
scripts/
  gaze_sim.py          v1: Gaze-LLE on the whole frame
  gaze_sim_v2.py       v2: per-person crop, ViT-L, smoothing
  promo/               demo video pipeline
    extract_v2.py      pose, silhouettes and gaze per frame
    depth.py           metric depth + field of view
    render_promo.py    attention rules + video composition
    audio.py           synthesized soundtrack
  blender/             3D store and a shopper's view (Blender)
    calib.py           camera + fixtures fitted to corners marked on the image
    layout.py          store plan, zones and fixture geometry
    overlay.py         draws the fitted fixtures over the image (check)
    trajectories.py    each person's feet and gaze ray, frame by frame
    store.py           builds the store in Blender and renders views
    warp.py            applies the lens distortion to compare with the image
    pov.py             renders what one shopper saw
    compose.py         side-by-side and shopper's-view videos
    shots.py           3D shots for the LinkedIn video (Blender)
    promo.py           vertical LinkedIn video (4:5)
    promo_audio.py     its synthesized soundtrack
config/
  zones_acti.json      store zones (polygons)
  camera_acti.json     calibrated camera (calib.py)
  layout_acti.json     fitted fixtures (calib.py)
docs/
  arquitetura.md       infrastructure, flow, libraries and licenses
  research.md          models, datasets and reference projects
  experiments/         one page per run, with numbers and learnings
data/videos/SOURCES.md source and license of each test video
```

Videos, extracted data and outputs are kept out of git (see `.gitignore`) and are generated locally.

## Limitations

- **No ground truth:** accuracy was not measured against manual annotations. Improvement between versions was judged by eye and through indirect metrics.
- **The 3D comes from a single image.** It is the surface seen from one viewpoint, so the virtual camera's orbit is small.
- The zones and the staff area were **drawn by hand** for this camera.
- **Not real time:** about 3 frames per second on an M2 Pro.
- **The shopper's view is a reconstruction.** His position is off by about 20 to 30 cm, the mannequins glide without moving their legs, and the ceiling, storefront and products are made up.

## Credits and licenses

BrandPose's code is released under the [MIT License](LICENSE). Third-party libraries, model weights and footage keep their own licenses, listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

- Video footage: public demo of the ACTi E43A camera, [on YouTube](https://www.youtube.com/watch?v=-1bRhYjw1qE). Footage © ACTi Corporation.
- [Gaze-LLE](https://github.com/fkryan/gazelle) (MIT) and [DINOv2](https://github.com/facebookresearch/dinov2) (Apache-2.0).
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (**AGPL-3.0**).
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Metric Indoor Base (the Base weights are **CC-BY-NC-4.0**, non-commercial use).
- [Blender](https://www.blender.org) (GPL-2.0-or-later), used as a tool to model and render the 3D store. It is not distributed here.
- Full list in [`docs/arquitetura.md`](docs/arquitetura.md#8-licenças-e-direitos-atenção-para-uso-comercial) (in Portuguese).
