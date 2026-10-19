"""Aplica a distorção da lente (calib.py) ao render sem distorção da câmera CCTV,
para casar pixel a pixel com a foto, e monta as comparações.

Grava OUT_PREFIX.jpg (o render com a lente, 1280x720), OUT_PREFIX_sobre_foto.jpg
(metade de cada) e OUT_PREFIX_lado_a_lado.jpg.

Uso: python warp.py RAW_RENDER CLEAN_PLATE OUT_PREFIX
"""

import argparse

import cv2
import numpy as np

import calib

OW, OH = 1280, 720


def distort(raw, cam, size=(OW, OH)):
    """Render sem distorção (com a sobra de calib.OVERSCAN) -> imagem com a lente da CCTV."""
    hr, wr = raw.shape[:2]
    W, H = calib.W, calib.H
    u, v = np.meshgrid(np.arange(size[0]) * W / size[0], np.arange(size[1]) * H / size[1])
    xd, yd = (u - W / 2) / cam["f"], (v - H / 2) / cam["f"]
    xn, yn = xd.copy(), yd.copy()
    for _ in range(20):
        k = 1 + cam["k1"] * (xn * xn + yn * yn)
        xn, yn = xd / k, yd / k
    s = (wr / 2) / calib.OVERSCAN
    out = cv2.remap(raw, (wr / 2 + xn * s).astype(np.float32), (hr / 2 + yn * s).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out[:, :, :3] if out.ndim == 3 and out.shape[2] == 4 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_render", help="cctv_raw.png do store.py")
    ap.add_argument("clean_plate", help="clean_plate.png (a loja sem pessoas)")
    ap.add_argument("out_prefix")
    args = ap.parse_args()

    warped = distort(cv2.imread(args.raw_render, cv2.IMREAD_UNCHANGED), calib.load_camera())
    cv2.imwrite(args.out_prefix + ".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ph = cv2.resize(cv2.imread(args.clean_plate), (OW, OH))
    cv2.imwrite(args.out_prefix + "_sobre_foto.jpg", cv2.addWeighted(ph, 0.5, warped, 0.5, 0))
    side = np.hstack([ph, warped])
    cv2.imwrite(args.out_prefix + "_lado_a_lado.jpg", cv2.resize(side, (OW * 2 * 3 // 4, OH * 3 // 4)),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
