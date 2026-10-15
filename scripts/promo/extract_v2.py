"""Extrai os dados por quadro para o vídeo promocional, com a configuração da V2.

- Pose + rastreamento (YOLO11m-pose + ByteTrack) em todos os quadros.
- Silhueta das pessoas (YOLO11m-seg) em todos os quadros, em meia resolução.
- Olhar (Gaze-LLE ViT-L, recorte por pessoa, supressão da própria cabeça)
  a cada `--gaze-every` quadros. O mapa de calor de cada pessoa é guardado em
  1/16 da resolução para a suavização ser feita na renderização.

Uso: python extract_v2.py VIDEO OUT_DIR [--gaze-every 3] [--max-frames N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaze_sim_v2 import crop_window, head_box  # noqa: E402

HM_DS = 16  # mapas de calor guardados em 1/16 da resolução


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--gaze-every", type=int, default=3)
    ap.add_argument("--model", default="gazelle_dinov2_vitl14_inout")
    ap.add_argument("--crop-scale", type=float, default=10.0)
    ap.add_argument("--crop-min", type=int, default=448)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pose = YOLO("yolo11m-pose.pt")
    seg = YOLO("yolo11m-seg.pt")
    gaze_model, transform = torch.hub.load("fkryan/gazelle", args.model, trust_repo=True)
    gaze_model.eval().to(device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, masks, gaze, hms = [], [], [], []
    idx, t0 = 0, time.time()
    W = H = None

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        H, W = frame.shape[:2]

        # Pose + rastreamento em todos os quadros
        r = pose.track(frame, persist=True, verbose=False, classes=[0], conf=0.3, device=device)[0]
        people = []
        if r.boxes is not None and len(r.boxes) and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            kp = r.keypoints.xy.cpu().numpy()
            kc = r.keypoints.conf.cpu().numpy()
            for tid, b, k, c in zip(ids, boxes, kp, kc):
                hb = head_box(k, c, b, W, H)
                people.append({
                    "id": tid,
                    "box": [round(float(v), 1) for v in b],
                    "kp": k.round(1).tolist(),
                    "kc": c.round(2).tolist(),
                    "head": None if hb is None else [round(float(v), 1) for v in hb],
                })
        frames.append(people)

        # Silhueta das pessoas (união), em meia resolução
        s = seg.predict(frame, verbose=False, classes=[0], conf=0.3, device=device)[0]
        m = np.zeros((H // 2, W // 2), np.uint8)
        if s.masks is not None:
            for poly in s.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(m, [np.round(poly / 2).astype(np.int32)], 1)
        masks.append(np.packbits(m))

        # Olhar, com a configuração da V2 (recorte por pessoa + ViT-L)
        valid = [p for p in people if p["head"] is not None]
        if idx % args.gaze_every == 0 and valid:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensors, bbs, wins = [], [], []
            for p in valid:
                x1, y1, x2, y2 = p["head"]
                left, top, side = crop_window((x1, y1, x2, y2), W, H, args.crop_scale, args.crop_min)
                tensors.append(transform(img.crop((left, top, left + side, top + side))))
                bbs.append([(max(0, (x1 - left) / side), max(0, (y1 - top) / side),
                             min(1, (x2 - left) / side), min(1, (y2 - top) / side))])
                wins.append((left, top, side))
            with torch.no_grad():
                o = gaze_model({"images": torch.stack(tensors).to(device), "bboxes": bbs})
            hs, ws = H // HM_DS, W // HM_DS
            for i, (p, (left, top, side)) in enumerate(zip(valid, wins)):
                hm = o["heatmap"][i][0].float().cpu().numpy()
                small = np.zeros((hs, ws), np.float32)
                l, t, sd = left // HM_DS, top // HM_DS, max(1, side // HM_DS)
                patch = cv2.resize(hm, (sd, sd), interpolation=cv2.INTER_AREA)
                small[t:t + sd, l:l + sd] = patch[: hs - t, : ws - l]
                # Suprime a própria cabeça (ampliada 1,5x) antes de escolher o pico
                x1, y1, x2, y2 = p["head"]
                cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.75, (y2 - y1) * 0.75
                pick = small.copy()
                pick[int(max(0, cy - hh) / HM_DS):int((cy + hh) / HM_DS) + 1,
                     int(max(0, cx - hw) / HM_DS):int((cx + hw) / HM_DS) + 1] = 0
                py, px = np.unravel_index(pick.argmax(), pick.shape)
                gaze.append({
                    "frame": idx, "id": p["id"], "inout": round(float(o["inout"][i][0]), 3),
                    "xy": [round((px + 0.5) * HM_DS), round((py + 0.5) * HM_DS)],
                    "peak": round(float(pick.max()), 4), "hm": len(hms),
                })
                hms.append(small.astype(np.float16))

        idx += 1
        if idx % 100 == 0:
            print(f"{idx} quadros, {len(gaze)} olhares, {time.time() - t0:.0f}s", flush=True)

    cap.release()
    meta = {"video": args.video, "fps": fps, "width": W, "height": H, "frames": idx,
            "gaze_every": args.gaze_every, "model": args.model, "hm_ds": HM_DS,
            "mask_shape": [H // 2, W // 2], "crop_scale": args.crop_scale, "crop_min": args.crop_min}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "tracks.json").write_text(json.dumps(frames))
    (out / "gaze.json").write_text(json.dumps(gaze))
    np.save(out / "gaze_heatmaps.npy", np.stack(hms) if hms else np.zeros((0, 1, 1), np.float16))
    np.save(out / "person_masks_packed.npy", np.stack(masks))
    print(json.dumps({"frames": idx, "gaze": len(gaze), "tracks": len({p['id'] for f in frames for p in f}),
                      "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
