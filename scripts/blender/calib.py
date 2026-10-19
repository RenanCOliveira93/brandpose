"""Calibra a câmera junto com os móveis, por mínimos quadrados.

A calibração anterior (depth.py) tirava altura e campo de visão do mapa de
profundidade, e a escala dele estava errada: as gôndolas saíam com 0,7 a 1,0 m.
Aqui a câmera (altura, giro, inclinação, rolagem, distância focal e distorção
radial da lente) é ajustada junto com os móveis, a partir de cantos marcados na
foto. A escala vem das pessoas: o nariz fica ~1,49 m acima do tornozelo.

O ajuste parte da câmera em config/camera_acti.json, se existir; senão, de CAM0.
A calibração de config/ foi construída aos poucos: a cada leva de pontos marcados,
o ajuste partiu do resultado anterior. Do zero, a partir de CAM0, ele cai num
mínimo local pior (câmera a 2,31 m, geladeiras fora do lugar, custo 2,1 vezes maior).

As coordenadas do mundo são as da loja (layout.py): x ao longo das gôndolas,
y atravessando, z para cima; a câmera fica sobre a origem.

Os outros scripts importam daqui o modelo da câmera (project, unproject).

Uso: python calib.py DATA_DIR  ->  config/camera_acti.json e config/layout_acti.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import layout

W, H = 1920, 1080
NOSE_OVER_ANKLE = 1.49  # pessoa de ~1,70 m
OVERSCAN = 0.70  # render da CCTV sem distorção: meia largura do quadro (tan), cobre os cantos da foto
# ponto de partida: a câmera que saiu da profundidade (depth.py), ~2 m e 60°, sem distorção
CAM0 = {"h": 2.0, "yaw": math.radians(32.5), "pitch": math.radians(18.0), "roll": 0.0,
        "f": W / 2 / math.tan(math.radians(30)), "k1": 0.0}

# zona: (parâmetros livres, observações). Ponto local: x ao longo da face (0 ou "L"),
# y para dentro do corpo (0 ou "d"), z para cima (0 ou "H"). "line": o pixel cai na reta.
OBS = {
    "DOCES": ("x y L H", [
        ("pt", (450, 444), (0, 0, "H")), ("pt", (1037, 275), ("L", 0, "H")),
        ("line", (631, 844), (0, 0, 0), ("L", 0, 0)),
        ("line", (619, 800), (0, 0, 0.15), ("L", 0, 0.15)), ("line", (925, 625), (0, 0, 0.15), ("L", 0, 0.15)),
    ]),
    "MERCEARIA": ("x y L d H", [
        ("pt", (1250, 650), (0, 0, "H")), ("pt", (1056, 587), (0, "d", "H")),
        ("pt", (1506, 356), ("L", 0, "H")),
        ("line", (1287, 987), (0, 0, 0), ("L", 0, 0)), ("line", (1431, 775), (0, 0, 0), ("L", 0, 0)),
    ]),
    "PONTA DE GÔNDOLA": ("x y L H", [
        ("pt", (880, 985), (0, 0, 0)), ("pt", (1035, 1040), ("L", 0, 0)), ("pt", (875, 720), (0, 0, "H")),
        ("pt", (1050, 760), ("L", 0, "H")),
    ]),
    "SNACKS": ("x y d H", [
        ("pt", (603, 197), (0, 0, "H")), ("pt", (431, 225), (0, "d", "H")),
        ("pt", (603, 412), (0, 0, 0)), ("pt", (431, 400), (0, "d", 0)),
    ]),
    "GELADEIRAS": ("x y L", [
        ("pt", (812, 18), (0, 0, "H")), ("pt", (1420, 30), ("L", 0, "H")),
        ("line", (1017, 231), (0, 0, 0.2), ("L", 0, 0.2)), ("line", (1324, 206), (0, 0, 0.2), ("L", 0, 0.2)),
    ]),
    "CERVEJAS": ("x y H", [
        ("pt", (1600, 431), (0, 0, 0)), ("pt", (1600, 117), (0, 0, "H")),
        ("line", (1805, 475), (0, 0, 0), ("L", 0, 0)), ("line", (1742, 144), (0, 0, "H"), ("L", 0, "H")),
    ]),
    "VINHOS": ("x y L H", [
        ("pt", (1505, 575), (0, 0, 0)), ("pt", (1592, 575), ("L", 0, 0)), ("pt", (1505, 219), (0, 0, "H")),
        ("pt", (1592, 219), ("L", 0, "H")),
    ]),
    "CAFÉ": ("x y a L", [
        ("pt", (300, 206), (0, 0, "H")), ("pt", (475, 200), ("L", 0, "H")),
    ]),
    "_EXPOSITOR": ("x y a L d H", [
        ("pt", (355, 457), (0, 0, "H")), ("pt", (550, 505), ("L", 0, "H")), ("pt", (550, 887), ("L", 0, 0)),
        ("pt", (620, 860), ("L", "d", 0)),
    ]),
    "_BEBIDAS": ("x y L H", [
        ("pt", (1042, 232), (0, 0, "H")), ("pt", (1225, 215), ("L", 0, "H")), ("pt", (1050, 485), (0, 0, 0)),
    ]),
    "_AGUA": ("x y a L H", [
        ("pt", (328, 306), (0, 0, "H")), ("pt", (426, 302), ("L", 0, "H")), ("pt", (334, 417), (0, 0, 0)),
        ("pt", (424, 408), ("L", 0, 0)),
    ]),
    "_SORVETE": ("x y a L H", [
        ("pt", (220, 274), (0, 0, "H")), ("pt", (278, 276), ("L", 0, "H")), ("pt", (222, 320), (0, 0, 0)),
        ("pt", (278, 322), ("L", 0, 0)),
    ]),
    "ÓCULOS": ("x y", [
        ("pt", (150, 625), (0, 0, 0)), ("pt", (142, 212), (0, 0, "H")),
    ]),
}
CAM_KEYS = ["h", "yaw", "pitch", "roll", "f", "k1"]


def load_camera():
    return json.loads(layout.CAMERA_FILE.read_text())


def cam_matrix(yaw, pitch, roll):
    """Colunas: direita, baixo e frente da câmera (convenção OpenCV) no mundo."""
    fw = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    rt = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    dn = np.cross(fw, rt)
    c, s = math.cos(roll), math.sin(roll)
    return np.stack([c * rt + s * dn, -s * rt + c * dn, fw], 1)


def project(cam, p):
    M = cam_matrix(cam["yaw"], cam["pitch"], cam["roll"])
    pc = M.T @ (np.asarray(p, float) - [0, 0, cam["h"]])
    xn, yn = pc[0] / pc[2], pc[1] / pc[2]
    k = 1 + cam["k1"] * (xn * xn + yn * yn)
    return np.array([cam["f"] * xn * k + W / 2, cam["f"] * yn * k + H / 2])


def unproject(cam, u, v, z):
    """Pixel -> ponto na altura z (inverte a distorção por iteração)."""
    xd, yd = (u - W / 2) / cam["f"], (v - H / 2) / cam["f"]
    xn, yn = xd, yd
    for _ in range(20):
        k = 1 + cam["k1"] * (xn * xn + yn * yn)
        xn, yn = xd / k, yd / k
    M = cam_matrix(cam["yaw"], cam["pitch"], cam["roll"])
    ray = M @ np.array([xn, yn, 1.0])
    t = (z - cam["h"]) / ray[2]
    return np.array([0, 0, cam["h"]]) + t * ray


def fixture_point(p, local):
    x = {"L": p["L"]}.get(local[0], local[0])
    y = {"d": p["d"]}.get(local[1], local[1])
    z = {"H": p["H"]}.get(local[2], local[2])
    a = math.radians(p["a"])
    return (p["x"] + math.cos(a) * x - math.sin(a) * y, p["y"] + math.sin(a) * x + math.cos(a) * y, z)


def people_samples(data):
    """Pessoas inteiras e nítidas, a cada 20 quadros: (quadro, id, nariz, tornozelos) em pixels."""
    tracks = json.loads((data / "tracks.json").read_text())
    out = []
    for fr in range(0, len(tracks), 20):
        for p in tracks[fr]:
            x0, y0, x1, y1 = p["box"]
            kc = p["kc"]
            if x0 < 5 or x1 > W - 5 or y1 > H - 5 or kc[0] < 0.8 or min(kc[15], kc[16]) < 0.6:
                continue
            ankle = np.mean([p["kp"][15], p["kp"][16]], 0)
            out.append((fr, p["id"], np.array(p["kp"][0]), ankle))
    return out


def main():
    from scipy.optimize import least_squares

    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="dados do vídeo (extract_v2.py)")
    args = ap.parse_args()

    cam0 = dict(CAM0)
    if layout.CAMERA_FILE.exists():  # refazendo (cantos novos, por exemplo): parte da calibração atual
        cam0 = {k: v for k, v in load_camera().items() if k in CAM_KEYS}
    fixtures = {z: {"x": st[0], "y": st[1], "a": a, "L": L, "d": d, "H": Hh}
                for z, kind, st, a, L, d, Hh in layout.INITIAL if z in OBS}
    for z, p in fixtures.items():  # cada móvel começa onde o raio do primeiro canto bate na altura dele
        o = next(o for o in OBS[z][1] if o[0] == "pt" and o[2][:2] == (0, 0))
        hit = unproject(cam0, *o[1], p["H"] if o[2][2] == "H" else o[2][2])
        p["x"], p["y"] = hit[0], hit[1]
    people = people_samples(args.data_dir)

    # vetor: câmera, móveis (só os livres), pessoas (x, y no piso, começando pelo raio dos tornozelos)
    keys = [("cam", k) for k in CAM_KEYS]
    for z, (free, _) in OBS.items():
        keys += [(z, k) for k in free.split()]
    x0 = [cam0[k] if g == "cam" else fixtures[g][k] for g, k in keys]
    for _, _, nose, ankle in people:
        x0 += list(unproject(cam0, *ankle, 0.08)[:2])

    def unpack(v):
        cam, fx = dict(cam0), {z: dict(p) for z, p in fixtures.items()}
        for (g, k), val in zip(keys, v):
            (cam if g == "cam" else fx[g])[k] = val
        return cam, fx, np.reshape(v[len(keys):], (-1, 2))

    def residuals(v, detail=False):
        cam, fx, pp = unpack(v)
        res, info = [], []
        for z, (_, obs) in OBS.items():
            for o in obs:
                if o[0] == "pt":
                    r = project(cam, fixture_point(fx[z], o[2])) - o[1]
                    res += list(r)
                    info.append((z, o[1], float(np.hypot(*r))))
                else:
                    a, b = project(cam, fixture_point(fx[z], o[2])), project(cam, fixture_point(fx[z], o[3]))
                    n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
                    r = float((np.array(o[1]) - a) @ n)
                    res.append(r)
                    info.append((z, o[1], abs(r)))
            for k in ("L", "H", "d"):
                res.append(50 * max(0.0, 0.2 - fx[z][k]))
        for (fr, pid, nose, ankle), (px, py) in zip(people, pp):
            res += list((project(cam, (px, py, 0.08)) - ankle) * 0.5)
            res += list((project(cam, (px, py, 0.08 + NOSE_OVER_ANKLE)) - nose) * 0.5)
        return (res, info) if detail else res

    r = least_squares(residuals, x0, loss="soft_l1", f_scale=8.0)
    cam, fx, _ = unpack(r.x)
    res, info = residuals(r.x, detail=True)
    print("câmera:", {k: round(math.degrees(v), 2) if k in ("yaw", "pitch", "roll") else round(float(v), 4)
                      for k, v in cam.items()},
          f"hfov={math.degrees(2 * math.atan(W / 2 / cam['f'])):.1f}°", f"pessoas={len(people)}")
    for z, p in fx.items():
        errs = [e for zz, _, e in info if zz == z]
        print(f"{z:18s} " + " ".join(f"{k}={v:6.2f}" for k, v in p.items()) + f"   erro médio={np.mean(errs):5.1f} px")
    print("piores:", sorted(info, key=lambda i: -i[2])[:6])
    M = cam_matrix(cam["yaw"], cam["pitch"], cam["roll"])
    cam = {k: float(v) for k, v in cam.items()}
    layout.CAMERA_FILE.write_text(json.dumps({**cam, "M": M.tolist()}, indent=1))
    layout.LAYOUT_FILE.write_text(json.dumps(
        {z: {k: round(float(v), 3) for k, v in p.items()} for z, p in fx.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
