"""Desenha o contorno 3D de cada móvel sobre a foto, pela câmera calibrada (calib.py).

Serve para conferir a calibração. Os círculos são os cantos marcados à mão
(brancos: o canto; amarelos: um ponto qualquer da aresta).

Uso: python overlay.py DATA_DIR OUT_IMAGE
"""

import argparse
from pathlib import Path

import cv2

import calib
import layout

COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 200, 0), (255, 0, 255), (255, 255, 255),
          (0, 128, 255), (255, 0, 0), (128, 255, 128), (200, 200, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="dados do vídeo (extract_v2.py), com clean_plate.png")
    ap.add_argument("out_image", type=Path)
    args = ap.parse_args()
    cam = calib.load_camera()

    def px(p):
        return tuple(int(v) for v in calib.project(cam, p))

    def edge(img, a, b, col, w):
        """Arestas em 12 pedaços: com a distorção da lente, retas viram curvas."""
        pts = [px([a[k] + (b[k] - a[k]) * i / 12 for k in range(3)]) for i in range(13)]
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, col, w)

    img = cv2.imread(str(args.data_dir / "clean_plate.png"))
    for i, (zone, kind, start, angle, L, d, H) in enumerate(layout.FIXTURES):
        c = layout.box_corners(start, angle, L, d, H)
        col = COLORS[i % len(COLORS)]
        for k in range(4):
            edge(img, c[k], c[(k + 1) % 4], col, 2)
            edge(img, c[4 + k], c[4 + (k + 1) % 4], col, 2)
            edge(img, c[k], c[4 + k], col, 1)
        cv2.putText(img, zone, px(c[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(img, zone, px(c[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    for obs in calib.OBS.values():
        for o in obs[1]:
            cv2.circle(img, o[1], 7, (0, 0, 0), 3)
            cv2.circle(img, o[1], 7, (255, 255, 255) if o[0] == "pt" else (0, 255, 255), 2)
    args.out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_image), img)


if __name__ == "__main__":
    main()
