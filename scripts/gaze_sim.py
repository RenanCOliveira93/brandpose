"""Simulação: para onde as pessoas olham num vídeo de loja.

Pipeline: YOLO-pose (pessoas + rastreamento + caixa da cabeça) -> Gaze-LLE
(mapa de calor do alvo do olhar) -> vídeo anotado + mapa de atenção acumulado
+ atenção por região (grade) da imagem.

Uso: python gaze_sim.py VIDEO OUT_DIR [--fps 3] [--max-seconds 30] [--grid 4x3]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Índices COCO dos pontos da cabeça e dos ombros
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
KP_CONF = 0.4


def head_box(kpts, kconf, person_box):
    """Estima a caixa da cabeça a partir dos pontos do rosto e dos ombros."""
    head_idx = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kconf[i] > KP_CONF]
    if not head_idx:
        # Sem pontos do rosto (de costas, longe, desfocado): usa o topo da caixa da pessoa
        x1, y1, x2, y2 = person_box
        if (y2 - y1) < 1.6 * (x2 - x1):  # pessoa cortada/sentada: não dá para chutar a cabeça
            return None
        size = min(x2 - x1, 0.16 * (y2 - y1))
        cx = (x1 + x2) / 2
        return (cx - size / 2, y1, cx + size / 2, y1 + size)
    pts = kpts[head_idx]
    cx, cy = pts.mean(axis=0)
    spread = np.ptp(pts, axis=0).max() if len(pts) > 1 else 0
    x1, y1, x2, y2 = person_box
    size = max(spread * 1.8, 0.12 * (y2 - y1))
    if kconf[L_SH] > KP_CONF and kconf[R_SH] > KP_CONF:
        size = max(size, 0.6 * np.linalg.norm(kpts[L_SH] - kpts[R_SH]))
    size = min(size, x2 - x1)
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--grid", default="4x3")
    ap.add_argument("--width", type=int, default=1280, help="redimensiona frames para esta largura")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    gaze_model, transform = torch.hub.load("fkryan/gazelle", "gazelle_dinov2_vitb14_inout", trust_repo=True)
    gaze_model.eval().to(device)
    pose = YOLO("yolo11m-pose.pt")

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / args.fps))
    max_frames = int(args.max_seconds * src_fps)

    writer = None
    accum = None
    first_frame = None
    gx, gy = (int(v) for v in args.grid.split("x"))
    grid_counts = np.zeros((gy, gx))
    log = []
    idx = 0

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        h0, w0 = frame.shape[:2]
        if w0 > args.width:
            frame = cv2.resize(frame, (args.width, round(h0 * args.width / w0)))
        H, W = frame.shape[:2]
        if accum is None:
            accum = np.zeros((H, W), np.float32)
            first_frame = frame.copy()
            writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

        res = pose.track(frame, persist=True, verbose=False, classes=[0], conf=0.3, device=device)[0]
        heads, ids = [], []
        if res.keypoints is not None and len(res.boxes):
            kp = res.keypoints.xy.cpu().numpy()
            kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kp.shape[:2])
            boxes = res.boxes.xyxy.cpu().numpy()
            tids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else list(range(len(boxes)))
            for k, c, b, t in zip(kp, kc, boxes, tids):
                hb = head_box(k, c, b)
                if hb is not None:
                    heads.append(hb)
                    ids.append(t)

        vis = frame.copy()
        if heads:
            norm = [(max(0, x1) / W, max(0, y1) / H, min(W, x2) / W, min(H, y2) / H) for x1, y1, x2, y2 in heads]
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                o = gaze_model({"images": transform(img).unsqueeze(0).to(device), "bboxes": [norm]})
            hms = o["heatmap"][0].float().cpu().numpy()  # [n, 64, 64]
            inouts = o["inout"][0].float().cpu().numpy()  # [n]
            overlay = np.zeros((H, W), np.float32)
            for (x1, y1, x2, y2), hm, io, tid in zip(heads, hms, inouts, ids):
                hm_full = cv2.resize(hm, (W, H))
                py, px = np.unravel_index(hm.argmax(), hm.shape)
                px, py = (px + 0.5) * W / 64, (py + 0.5) * H / 64
                hc = ((x1 + x2) / 2, (y1 + y2) / 2)
                log.append({"frame": idx, "t": round(idx / src_fps, 2), "track": tid, "inout": round(float(io), 3),
                            "gaze_xy": [round(px), round(py)], "head_xyxy": [round(v) for v in (x1, y1, x2, y2)]})
                color = (0, 255, 255) if io > 0.5 else (128, 128, 128)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(vis, f"#{tid} in={io:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if io > 0.5:
                    overlay = np.maximum(overlay, hm_full)
                    accum += hm_full * io
                    gi, gj = min(gy - 1, int(py / H * gy)), min(gx - 1, int(px / W * gx))
                    grid_counts[gi, gj] += 1
                    cv2.arrowedLine(vis, (int(hc[0]), int(hc[1])), (int(px), int(py)), (0, 0, 255), 2, tipLength=0.04)
                    cv2.circle(vis, (int(px), int(py)), 8, (0, 0, 255), -1)
            color_hm = cv2.applyColorMap((overlay * 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (overlay > 0.15)[..., None]
            vis = np.where(mask, cv2.addWeighted(vis, 0.55, color_hm, 0.45, 0), vis)
        cv2.putText(vis, f"t={idx / src_fps:.1f}s  pessoas c/ cabeca: {len(heads)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(vis)
        if len(log) and idx // step % max(1, int(args.fps * 2)) == 0:
            cv2.imwrite(str(out / f"frame_{idx:05d}.jpg"), vis)
        idx += 1

    cap.release()
    if writer:
        writer.release()

    # Mapa de atenção acumulado sobre o primeiro frame (só faz sentido com câmera fixa)
    if accum is not None and accum.max() > 0:
        a = accum / accum.max()
        hm = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blend = np.where((a > 0.05)[..., None], cv2.addWeighted(first_frame, 0.5, hm, 0.5, 0), first_frame)
        H, W = a.shape
        for i in range(1, gx):
            cv2.line(blend, (i * W // gx, 0), (i * W // gx, H), (255, 255, 255), 1)
        for j in range(1, gy):
            cv2.line(blend, (0, j * H // gy), (W, j * H // gy), (255, 255, 255), 1)
        total = grid_counts.sum()
        for j in range(gy):
            for i in range(gx):
                pct = 100 * grid_counts[j, i] / total if total else 0
                cv2.putText(blend, f"{pct:.0f}%", (i * W // gx + 8, j * H // gy + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / "attention_map.jpg"), blend)

    summary = {
        "video": args.video, "sampled_fps": args.fps, "observations": len(log),
        "in_frame_observations": int(grid_counts.sum()), "tracks": len({r["track"] for r in log}),
        "grid": args.grid, "grid_pct": (100 * grid_counts / max(1, grid_counts.sum())).round(1).tolist(),
    }
    (out / "gaze_log.json").write_text(json.dumps(log, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
