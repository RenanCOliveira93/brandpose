"""Simulação v2: para onde as pessoas olham num vídeo de loja.

Mudanças em relação ao gaze_sim.py (v1):
- `--crop`: roda o Gaze-LLE num recorte quadrado em volta de cada cabeça, em vez
  do quadro inteiro espremido para 448x448. A cabeça chega ao modelo maior e sem
  distorção de proporção.
- `--model`: permite trocar a variante do Gaze-LLE (ex.: ViT-L).
- `--smooth`: média móvel exponencial do mapa de calor por pessoa (track).
- `--suppress-self`: ignora o pico do mapa que cai na própria cabeça.
- Não chuta cabeça de pessoa cortada pela borda do quadro.
- Métricas extras no summary: jitter do olhar e % do olhar na própria cabeça.

Uso: python gaze_sim_v2.py VIDEO OUT_DIR [--crop] [--model ...] [--smooth 0.5] [--suppress-self]
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
SMOOTH_DS = 8  # mapas suavizados são guardados em 1/8 da resolução
TRACK_GAP_S = 1.0  # depois desse intervalo sem ver a pessoa, recomeça a média


def head_box(kpts, kconf, person_box, W, H):
    """Estima a caixa da cabeça a partir dos pontos do rosto e dos ombros."""
    x1, y1, x2, y2 = person_box
    head_idx = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kconf[i] > KP_CONF]
    if not head_idx:
        # Sem pontos do rosto: usa o topo da caixa, mas só se a pessoa estiver inteira
        cut = x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2
        if cut or (y2 - y1) < 1.6 * (x2 - x1):
            return None
        size = min(x2 - x1, 0.16 * (y2 - y1))
        cx = (x1 + x2) / 2
        return (cx - size / 2, y1, cx + size / 2, y1 + size)
    pts = kpts[head_idx]
    cx, cy = pts.mean(axis=0)
    spread = np.ptp(pts, axis=0).max() if len(pts) > 1 else 0
    size = max(spread * 1.8, 0.12 * (y2 - y1))
    if kconf[L_SH] > KP_CONF and kconf[R_SH] > KP_CONF:
        size = max(size, 0.6 * np.linalg.norm(kpts[L_SH] - kpts[R_SH]))
    size = min(size, x2 - x1)
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def crop_window(hb, W, H, scale, min_side):
    """Recorte quadrado em volta da cabeça. A cabeça fica no terço de cima,
    porque com câmera alta o alvo costuma estar abaixo dela."""
    x1, y1, x2, y2 = hb
    hw = max(x2 - x1, y2 - y1)
    side = int(np.clip(hw * scale, min_side, min(W, H)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + 0.2 * side
    left = int(np.clip(cx - side / 2, 0, W - side))
    top = int(np.clip(cy - side / 2, 0, H - side))
    return left, top, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--grid", default="4x3")
    ap.add_argument("--width", type=int, default=1920, help="redimensiona frames para esta largura")
    ap.add_argument("--model", default="gazelle_dinov2_vitb14_inout")
    ap.add_argument("--crop", action="store_true", help="roda o modelo num recorte por cabeça")
    ap.add_argument("--crop-scale", type=float, default=10.0, help="lado do recorte em larguras de cabeça")
    ap.add_argument("--crop-min", type=int, default=448)
    ap.add_argument("--smooth", type=float, default=0.0, help="peso do mapa anterior na média móvel (0 = sem)")
    ap.add_argument("--suppress-self", action="store_true", help="ignora o pico na própria cabeça")
    ap.add_argument("--track-every-frame", action="store_true", help="roda o tracker em todos os quadros")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    gaze_model, transform = torch.hub.load("fkryan/gazelle", args.model, trust_repo=True)
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
    track_state = {}  # track -> (t, mapa suavizado em 1/8)
    log = []
    idx = 0

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        sampled = idx % step == 0
        if not sampled and not args.track_every_frame:
            idx += 1
            continue
        h0, w0 = frame.shape[:2]
        if w0 > args.width:
            frame = cv2.resize(frame, (args.width, round(h0 * args.width / w0)))
        H, W = frame.shape[:2]
        t_now = idx / src_fps

        # Rastrear em todos os quadros evita que o tracker perca quem se move entre amostras
        res = pose.track(frame, persist=True, verbose=False, classes=[0], conf=0.3, device=device)[0]
        if not sampled:
            idx += 1
            continue
        if accum is None:
            accum = np.zeros((H, W), np.float32)
            first_frame = frame.copy()
            writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        heads, ids = [], []
        if res.keypoints is not None and len(res.boxes):
            kp = res.keypoints.xy.cpu().numpy()
            kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kp.shape[:2])
            boxes = res.boxes.xyxy.cpu().numpy()
            tids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else list(range(len(boxes)))
            for k, c, b, t in zip(kp, kc, boxes, tids):
                hb = head_box(k, c, b, W, H)
                if hb is not None:
                    heads.append(tuple(float(v) for v in hb))
                    ids.append(t)

        vis = frame.copy()
        if heads:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Mapa de calor de cada pessoa em coordenadas do quadro inteiro
            full_hms, inouts = [], []
            with torch.no_grad():
                if args.crop:
                    tensors, bbs, wins = [], [], []
                    for x1, y1, x2, y2 in heads:
                        left, top, side = crop_window((x1, y1, x2, y2), W, H, args.crop_scale, args.crop_min)
                        tensors.append(transform(img.crop((left, top, left + side, top + side))))
                        bbs.append([(max(0, (x1 - left) / side), max(0, (y1 - top) / side),
                                     min(1, (x2 - left) / side), min(1, (y2 - top) / side))])
                        wins.append((left, top, side))
                    o = gaze_model({"images": torch.stack(tensors).to(device), "bboxes": bbs})
                    for i, (left, top, side) in enumerate(wins):
                        hm = o["heatmap"][i][0].float().cpu().numpy()
                        canvas = np.zeros((H, W), np.float32)
                        canvas[top:top + side, left:left + side] = cv2.resize(hm, (side, side))
                        full_hms.append(canvas)
                        inouts.append(float(o["inout"][i][0]))
                else:
                    norm = [(max(0, x1) / W, max(0, y1) / H, min(W, x2) / W, min(H, y2) / H) for x1, y1, x2, y2 in heads]
                    o = gaze_model({"images": transform(img).unsqueeze(0).to(device), "bboxes": [norm]})
                    for hm, io in zip(o["heatmap"][0].float().cpu().numpy(), o["inout"][0].float().cpu().numpy()):
                        full_hms.append(cv2.resize(hm, (W, H)))
                        inouts.append(float(io))

            overlay = np.zeros((H, W), np.float32)
            for (x1, y1, x2, y2), hm_full, io, tid in zip(heads, full_hms, inouts, ids):
                small = cv2.resize(hm_full, (W // SMOOTH_DS, H // SMOOTH_DS), interpolation=cv2.INTER_AREA)
                if args.smooth > 0:
                    prev = track_state.get(tid)
                    if prev is not None and t_now - prev[0] <= TRACK_GAP_S:
                        small = args.smooth * prev[1] + (1 - args.smooth) * small
                    track_state[tid] = (t_now, small)
                pick = small.copy()
                if args.suppress_self:
                    # Zera a própria cabeça (ampliada 1,5x) antes de escolher o pico
                    hw, hh = (x2 - x1) * 0.75, (y2 - y1) * 0.75
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    sx1, sy1 = int(max(0, cx - hw) / SMOOTH_DS), int(max(0, cy - hh) / SMOOTH_DS)
                    sx2, sy2 = int(min(W, cx + hw) / SMOOTH_DS) + 1, int(min(H, cy + hh) / SMOOTH_DS) + 1
                    pick[sy1:sy2, sx1:sx2] = 0
                py, px = np.unravel_index(pick.argmax(), pick.shape)
                px, py = (px + 0.5) * SMOOTH_DS, (py + 0.5) * SMOOTH_DS
                self_hit = x1 <= px <= x2 and y1 <= py <= y2
                hm_draw = cv2.resize(small, (W, H))
                if hm_draw.max() > 0:
                    hm_draw = hm_draw / hm_draw.max()
                hc = ((x1 + x2) / 2, (y1 + y2) / 2)
                log.append({"frame": idx, "t": round(t_now, 2), "track": tid, "inout": round(io, 3),
                            "gaze_xy": [round(px), round(py)], "head_xyxy": [round(v) for v in (x1, y1, x2, y2)],
                            "self_hit": bool(self_hit)})
                color = (0, 255, 255) if io > 0.5 else (128, 128, 128)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(vis, f"#{tid} in={io:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if io > 0.5:
                    overlay = np.maximum(overlay, hm_draw)
                    accum += hm_draw * io
                    gi, gj = min(gy - 1, int(py / H * gy)), min(gx - 1, int(px / W * gx))
                    grid_counts[gi, gj] += 1
                    cv2.arrowedLine(vis, (int(hc[0]), int(hc[1])), (int(px), int(py)), (0, 0, 255), 2, tipLength=0.04)
                    cv2.circle(vis, (int(px), int(py)), 8, (0, 0, 255), -1)
            color_hm = cv2.applyColorMap((overlay * 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (overlay > 0.15)[..., None]
            vis = np.where(mask, cv2.addWeighted(vis, 0.55, color_hm, 0.45, 0), vis)
        cv2.putText(vis, f"t={t_now:.1f}s  pessoas c/ cabeca: {len(heads)}", (10, 24),
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
        "video": args.video, "args": vars(args), "observations": len(log),
        "in_frame_observations": int(grid_counts.sum()), "tracks": len({r["track"] for r in log}),
        "self_hit_pct": round(100 * sum(r["self_hit"] for r in log) / max(1, len(log)), 1),
        "jitter_px_median": jitter(log),
        "grid": args.grid, "grid_pct": (100 * grid_counts / max(1, grid_counts.sum())).round(1).tolist(),
    }
    (out / "gaze_log.json").write_text(json.dumps(log, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))


def jitter(log):
    """Mediana do deslocamento do ponto do olhar entre amostras seguidas da mesma pessoa."""
    by_track = {}
    for r in log:
        by_track.setdefault(r["track"], []).append(r)
    d = []
    for rs in by_track.values():
        for a, b in zip(rs, rs[1:]):
            d.append(float(np.hypot(a["gaze_xy"][0] - b["gaze_xy"][0], a["gaze_xy"][1] - b["gaze_xy"][1])))
    return round(float(np.median(d)), 1) if d else None


if __name__ == "__main__":
    main()
