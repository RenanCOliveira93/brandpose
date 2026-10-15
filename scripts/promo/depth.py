"""Profundidade métrica de quadros do vídeo (Depth Anything V2 Metric Indoor).

Gera depth_<nome>.npy (metros, 1080x1920) e escolhe o campo de visão horizontal
(a lente da câmera não é conhecida): o que deixa as faces das gôndolas DOCES e
MERCEARIA mais perpendiculares ao piso. Minimizar só o resíduo do plano do piso
não funciona: ele sempre melhora com campos de visão menores (a cena "achata").

Uso: python depth.py DATA_DIR --frame 570 [--model ...]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Áreas de piso visíveis na cena (para estimar o campo de visão)
FLOOR_POLYS = [
    [[190, 700], [330, 650], [360, 880], [560, 905], [630, 850], [870, 705], [870, 1075], [190, 1075]],
    [[1520, 640], [1760, 500], [1760, 900], [1560, 1000], [1500, 760]],
]


def predict_depth(model, proc, bgr, device):
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # [1, h, w]
    d = torch.nn.functional.interpolate(pred[:, None], size=bgr.shape[:2], mode="bicubic", align_corners=False)
    return d[0, 0].float().cpu().numpy().clip(0.05, None)


def backproject(depth, hfov_deg, ys, xs):
    H, W = depth.shape
    f = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    z = depth[ys, xs]
    return np.stack([(xs - W / 2) * z / f, (ys - H / 2) * z / f, z], axis=1)


def plane_residual(pts, iters=300, seed=0):
    """RANSAC simples: devolve (normal, d, rms dos inliers / profundidade média)."""
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    thr = 0.02 * pts[:, 2].mean()
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        r = np.abs(pts @ n + d)
        inl = r < thr
        if inl.sum() < 0.5 * len(pts):
            continue
        rms = np.sqrt((r[inl] ** 2).mean()) / pts[:, 2].mean()
        score = rms / inl.mean()
        if score < best[2]:
            best = (n, d, score)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--frame", type=int, required=True, help="quadro do congelamento (bullet time)")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf")
    args = ap.parse_args()

    D = Path(args.data_dir)
    meta = json.loads((D / "meta.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    assert ok
    cv2.imwrite(str(D / f"freeze_{args.frame}.png"), frame)
    plate = cv2.imread(str(D / "clean_plate.png"))

    d_freeze = predict_depth(model, proc, frame, device)
    d_plate = predict_depth(model, proc, plate, device)
    np.save(D / f"depth_freeze_{args.frame}.npy", d_freeze.astype(np.float32))
    np.save(D / "depth_plate.npy", d_plate.astype(np.float32))

    # Campo de visão que deixa as faces das gôndolas perpendiculares ao piso
    H, W = d_plate.shape
    zones = {z["short"]: z["polys"] for z in json.loads((D / "zones.json").read_text())["zones"]}

    def sample(polys, n=5000, erode=15):
        m = np.zeros((H, W), np.uint8)
        for p in polys:
            cv2.fillPoly(m, [np.array(p, np.int32)], 1)
        m = cv2.erode(m, np.ones((erode, erode), np.uint8))
        ys, xs = np.nonzero(m)
        sel = np.random.default_rng(0).choice(len(ys), min(n, len(ys)), replace=False)
        return ys[sel], xs[sel]

    floor = sample(FLOOR_POLYS, erode=1)
    faces = [sample(zones[k][:1]) for k in ("DOCES", "MERCEARIA")]
    scores = {}
    for hfov in range(50, 121, 5):
        n, dd, _ = plane_residual(backproject(d_plate, hfov, *floor))
        err = 0.0
        for ys, xs in faces:
            nk = plane_residual(backproject(d_plate, hfov, ys, xs))[0]
            err += 90.0 if nk is None else abs(90 - np.degrees(np.arccos(abs(n @ nk))))
        scores[hfov] = (err, n, dd)
    best = min(scores, key=lambda k: scores[k][0])
    n, dd = scores[best][1], scores[best][2]
    info = {
        "freeze_frame": args.frame, "model": args.model, "hfov_deg": best,
        "floor_normal": n.tolist(), "floor_d": float(dd),
        "fov_error_deg": {k: round(float(v[0]), 2) for k, v in scores.items()},
        "depth_freeze_range_m": [round(float(np.percentile(d_freeze, 2)), 2), round(float(np.percentile(d_freeze, 98)), 2)],
    }
    (D / "depth_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
