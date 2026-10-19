"""Monta os vídeos da câmera subjetiva (pov.py), no idioma pedido.

pov_<idioma>_lado_a_lado.mp4 (1920x1080, o corte para o YouTube): à esquerda, a
CCTV original com ele marcado e a linha do olhar (do olho ao alvo, pela câmera
calibrada); à direita, a visão dele.
pov_<idioma>_visao.mp4: só a visão dele, com a mira no centro e a zona que ele olha.
pov_<idioma>_limpo.mp4: só a visão dele, sem nada por cima.

Uso: python compose.py DATA_DIR RUN_DIR [--lang en] [--from F] [--to F]
  --from/--to: quadros do vídeo original (padrão: tudo o que o pov.py renderizou).
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import calib
import layout
from trajectories import MERGE

WHO = 12  # o rapaz de jaqueta azul
FPS = 30
W, H = 1920, 1080
PANEL = (960, 540)
TOP = 236  # topo dos painéis no lado a lado
YELLOW = (40, 210, 255)  # BGR; o mesmo amarelo do feixe do olhar no Blender
AUTHOR = "Claudio Escudero  ·  linkedin.com/in/claudioescudero"
TEXTS = {
    "pt": {
        "dec": ",", "title": "Pelos olhos de um cliente",
        "sub": "Uma câmera de segurança real e a visão dele, reconstruída em 3D no Blender",
        "cctv": "Câmera da loja", "pov": "O que ele via (reconstrução)", "him": "ele", "looking": "Olhando para: {}",
        "notes": ["A posição vem da câmera calibrada, e a direção, do olhar estimado pelo Gaze-LLE.",
                  "São estimativas: o trajeto erra cerca de 20 a 30 cm, e teto, vitrine e produtos são inventados."],
        "credit": "Imagens da câmera: demonstração pública da ACTi E43A  ·  © ACTi Corporation",
    },
    "en": {
        "dec": ".", "title": "Through a shopper's eyes",
        "sub": "A real security camera and his view, rebuilt in 3D in Blender",
        "cctv": "Store camera", "pov": "What he saw (reconstruction)", "him": "him", "looking": "Looking at: {}",
        "notes": ["Position comes from the calibrated camera, and direction from the gaze estimated by Gaze-LLE.",
                  "These are estimates: the path is off by about 20 to 30 cm, and the ceiling, storefront and products are made up."],
        "credit": "Camera footage: ACTi E43A public demo  ·  © ACTi Corporation",
    },
}


def font(size, weight="bold"):
    """Avenir Next, como no vídeo de demonstração; fora do macOS, a fonte padrão do Pillow."""
    try:
        index = {"bold": 0, "demi": 2, "medium": 5}[weight]
        return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", size, index=index)
    except OSError:
        return ImageFont.load_default(size)


def text(img, xy, s, size, fill=(255, 255, 255), anchor="la", box=None, weight="bold"):
    """Escreve com acentos (PIL) num quadro BGR do OpenCV."""
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f = font(size, weight)
    if box:
        l, t, r, b = d.textbbox(xy, s, font=f, anchor=anchor)
        d.rounded_rectangle((l - 12, t - 8, r + 12, b + 8), 8, fill=box)
    d.text(xy, s, font=f, fill=fill, anchor=anchor)
    img[:] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def mark(img, rec, persons, cam, label=None):
    """Marca o rapaz num quadro da CCTV (1920x1080): caixa e linha do olhar até o alvo em 3D.

    rec é o registro do quadro no pov_info.json; persons, as pessoas do quadro no tracks.json.
    Traços e textos grossos: o quadro vai para a tela reduzido."""
    me = next((p for p in persons if MERGE.get(p["id"], p["id"]) == WHO), None)
    eye = calib.project(cam, rec["eye"]).astype(int)
    if me and me.get("head"):  # a linha sai da cabeça detectada (o 3D erra uns 20-30 cm)
        hb = me["head"]
        eye = np.array([(hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2]).astype(int)
    tgt = calib.project(cam, rec["target"]).astype(int)
    cv2.line(img, tuple(eye), tuple(tgt), YELLOW, 6, cv2.LINE_AA)
    cv2.circle(img, tuple(tgt), 16, YELLOW, -1, cv2.LINE_AA)
    if me:
        x0, y0, x1, y1 = map(int, me["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), YELLOW, 6)
        if label:
            text(img, (max(x0, 0) + 14, max(y0 - 14, 60)), label, 44, fill=(20, 20, 20), anchor="ls",
                 box=(255, 210, 40))
    return img


def writer(path, size):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="dados do vídeo (extract_v2.py)")
    ap.add_argument("run_dir", type=Path, help="pasta da rodada, com pov_<idioma>/ do pov.py")
    ap.add_argument("--lang", choices=sorted(TEXTS), default="pt")
    ap.add_argument("--from", dest="first", type=int)
    ap.add_argument("--to", dest="last", type=int)
    args = ap.parse_args()
    tx, names = TEXTS[args.lang], layout.zone_names(args.lang)

    cam = calib.load_camera()
    meta = json.loads((args.data_dir / "meta.json").read_text())
    tracks = json.loads((args.data_dir / "tracks.json").read_text())
    pov = args.run_dir / f"pov_{args.lang}"
    info = json.loads((pov / "pov_info.json").read_text())
    k0 = (args.first or info[0]["frame"]) - info[0]["frame"]
    k1 = (args.last or info[-1]["frame"]) - info[0]["frame"]

    # zona olhada sem piscar: a mais frequente em ~0,4 s
    zones = [i["zone"] for i in info]
    steady = [Counter(zones[max(0, k - 6): k + 7]).most_common(1)[0][0] for k in range(len(zones))]

    # fundo do lado a lado: títulos, notas e créditos não mudam
    base = np.full((H, W, 3), (24, 20, 18), np.uint8)
    text(base, (W // 2, 74), tx["title"], 54, anchor="mm")
    text(base, (W // 2, 134), tx["sub"], 28, fill=(185, 195, 205), anchor="mm", weight="medium")
    text(base, (PANEL[0] // 2, TOP - 30), tx["cctv"], 26, fill=(230, 235, 240), anchor="mm", weight="demi")
    text(base, (PANEL[0] * 3 // 2, TOP - 30), tx["pov"], 26, fill=(230, 235, 240), anchor="mm", weight="demi")
    text(base, (24, TOP + PANEL[1] + 30), tx["credit"], 19, fill=(140, 150, 160), anchor="lm", weight="medium")
    for j, note in enumerate(tx["notes"]):
        text(base, (W // 2, TOP + PANEL[1] + 112 + 44 * j), note, 26, fill=(200, 208, 216), anchor="mm",
             weight="medium")
    text(base, (W // 2, H - 44), AUTHOR, 22, fill=(170, 180, 190), anchor="mm", weight="demi")

    def pov_frame(k):
        img = cv2.imread(str(pov / "frames" / f"{k + 1:04d}.png"))
        clean = img.copy()
        h, w = img.shape[:2]
        c = (w // 2, h // 2)
        cv2.circle(img, c, 18, YELLOW, 3, cv2.LINE_AA)
        cv2.circle(img, c, 3, YELLOW, -1, cv2.LINE_AA)
        if steady[k]:
            text(img, (24, h - 28), tx["looking"].format(names[steady[k]]), 30, anchor="ls", box=(20, 20, 20))
        return clean, img

    def cctv_frame(img, k):
        f = info[k]["frame"]
        mark(img, info[k], tracks[f], cam, tx["him"])
        text(img, (30, 66), f"{f / FPS:.1f} s".replace(".", tx["dec"]), 44, box=(20, 20, 20))
        return img

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, info[k0]["frame"])
    name = f"pov_{args.lang}"
    both = writer(args.run_dir / f"{name}_lado_a_lado.mp4", (W, H))
    solo = writer(args.run_dir / f"{name}_visao.mp4", (1280, 720))
    plain = writer(args.run_dir / f"{name}_limpo.mp4", (1280, 720))
    for k in range(k0, k1 + 1):
        ok, frame = cap.read()
        clean, view = pov_frame(k)
        plain.stdin.write(clean.tobytes())
        solo.stdin.write(view.tobytes())
        canvas = base.copy()
        canvas[TOP:TOP + PANEL[1], :PANEL[0]] = cv2.resize(cctv_frame(frame, k), PANEL, interpolation=cv2.INTER_AREA)
        canvas[TOP:TOP + PANEL[1], PANEL[0]:] = cv2.resize(view, PANEL, interpolation=cv2.INTER_AREA)
        both.stdin.write(canvas.tobytes())
    for p in (both, solo, plain):
        p.stdin.close()
        p.wait()
    print("ok", k1 - k0 + 1, "quadros")


if __name__ == "__main__":
    main()
