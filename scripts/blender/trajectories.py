"""Trajeto e olhar de cada pessoa, quadro a quadro, na câmera calibrada (calib.py).

Pé: tornozelos levados ao piso; se os pés saem do quadro, o nariz é levado à
altura de 1,57 m (de costas, o centro da cabeça; sem cabeça, os ombros a 1,40 m).
Olhar: raio da câmera pelo pico do Gaze-LLE (a cada 3 quadros); o Blender acha
onde ele bate na cena. Grava RUN_DIR/trajectories.json.

Uso: python trajectories.py DATA_DIR RUN_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

import calib

W, H = calib.W, calib.H
NOSE_Z = 0.08 + calib.NOSE_OVER_ANKLE
SHOULDER_Z = 1.40
# o rastreador trocou o número de algumas pessoas no meio do caminho (conferido no vídeo)
MERGE = {18: 12, 21: 12, 17: 8, 11: 15, 9: 15, 23: 14, 24: 14, 20: 4, 25: 4}


def ray(cam, u, v):
    far = calib.unproject(cam, u, v, -10.0)
    r = far - np.array([0, 0, cam["h"]])
    return (r / np.linalg.norm(r)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="dados do vídeo (extract_v2.py)")
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    cam = calib.load_camera()
    tracks = json.loads((args.data_dir / "tracks.json").read_text())
    gaze = {(g["frame"], g["id"]): g for g in json.loads((args.data_dir / "gaze.json").read_text())}
    people = {}
    for fr, persons in enumerate(tracks):
        for p in sorted(persons, key=lambda p: p["id"] in MERGE):  # o número original vem primeiro
            pid = MERGE.get(p["id"], p["id"])
            if people.get(pid) and people[pid][-1]["frame"] == fr:
                continue
            kc, kp, box = p["kc"], p["kp"], p["box"]
            if min(kc[15], kc[16]) > 0.5 and box[3] < H - 8:
                feet, src = calib.unproject(cam, *np.mean([kp[15], kp[16]], 0), 0.08), "tornozelo"
            elif kc[0] > 0.5 and box[1] > 8:
                feet, src = calib.unproject(cam, *kp[0], NOSE_Z), "nariz"
            elif p.get("head") and box[1] > 8:  # de costas: centro da cabeça
                hb = p["head"]
                feet, src = calib.unproject(cam, (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2, NOSE_Z + 0.03), "cabeça"
            elif min(kc[5], kc[6]) > 0.5 and box[1] > 8:
                feet, src = calib.unproject(cam, *np.mean([kp[5], kp[6]], 0), SHOULDER_Z), "ombros"
            else:
                continue
            rec = {"frame": fr, "feet": [round(float(feet[0]), 3), round(float(feet[1]), 3)], "src": src}
            g = gaze.get((fr, p["id"]))
            # mesma regra do projeto: pico a menos de 32 px da borda é "fora da imagem"
            if g and g["inout"] > 0.5 and min(g["xy"][0], g["xy"][1], W - g["xy"][0], H - g["xy"][1]) >= 32:
                rec["gaze_ray"] = ray(cam, *g["xy"])
                rec["gaze_xy"] = g["xy"]
            people.setdefault(pid, []).append(rec)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "trajectories.json").write_text(json.dumps({"fps": 30, "people": people}))
    for pid, recs in people.items():
        print(pid, recs[0]["frame"], recs[-1]["frame"], len(recs), "olhares", sum("gaze_ray" in r for r in recs))


if __name__ == "__main__":
    main()
