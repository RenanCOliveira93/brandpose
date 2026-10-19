"""Tomadas 3D do vídeo do LinkedIn, em pé (4:5, 1080x1350), com o tempo congelado.

Abre a cena do pov.py (RUN_DIR/pov_<idioma>/pov.blend: loja fechada, pessoas animadas
e a câmera dele), põe o rapaz como boneco e renderiza:
  cctv   o quadro de 18,5 s visto pela câmera de segurança (sem distorção, com sobra;
         o promo.py aplica a lente para casar com o vídeo e recorta o meio em 4:5);
  voo    a câmera sai da CCTV, dá a volta no rapaz e entra nos olhos dele: termina
         exatamente no primeiro quadro da visão dele a partir de 18,5 s;
  visao  o que ele vê, nos quadros que o promo.py usa (1x até 27 s, depois 1 a cada 3);
  mapa   no fim do trecho, sem teto: o caminho dele no piso, os pontos para onde
         olhou e o olhar daquele instante, numa câmera que desce em volta da loja (225 quadros).

Uso: Blender -b --python shots.py -- RUN_DIR [--lang en] [--shot voo] [--samples 32] [--every N] [--preview]
Grava em RUN_DIR/promo_<idioma>/: cctv_raw.png e voo/, voo_sem/, visao/, mapa/ (PNG, retomável).
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # o Blender não põe a pasta do script no caminho

import bpy
import numpy as np
from mathutils import Matrix, Vector

import calib
import store as S

ap = argparse.ArgumentParser(prog="shots.py")
ap.add_argument("run_dir", type=Path, help="pasta da rodada, com pov_<idioma>/pov.blend do pov.py")
ap.add_argument("--lang", choices=["pt", "en"], default="pt")
ap.add_argument("--shot", choices=["cctv", "voo", "visao", "mapa"], help="só esta tomada")
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--every", type=int, default=1, help="renderiza 1 a cada N quadros (prévia)")
ap.add_argument("--preview", action="store_true", help="Workbench, para acertar os movimentos de câmera")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])

FREEZE, F_FAST = 555, 810  # 18,5 s: o rapaz já está inteiro na imagem da CCTV; 27 s: a visão acelera
W, H = 1080, 1350  # em pé, como o vídeo
CX = 640  # centro do recorte 4:5 da CCTV, em pixels da imagem dela (o mesmo do promo.py)
EYE_FOV = 70  # campo de visão vertical dele, em graus (como um celular em pé)
HIM = ((0.30, 0.42, 0.65), (0.80, 0.60, 0.45), (0.03, 0.04, 0.10), (0.03, 0.025, 0.02))  # jaqueta azul-clara
pov_dir = args.run_dir / f"pov_{args.lang}"
out = args.run_dir / (f"promo_{args.lang}" + ("_previa" if args.preview else ""))  # a prévia não trava o render
info = json.loads((pov_dir / "pov_info.json").read_text())
F0, F1 = info[0]["frame"], info[-1]["frame"]

bpy.ops.wm.open_mainfile(filepath=str(pov_dir / "pov.blend"))
sc = bpy.context.scene
S.MATS.update({k: bpy.data.materials[n] for k, n in
               {"roupa": "Roupa", "pele": "Pele", "sapato": "Sapato", "cabelo": "Cabelo"}.items()})
S.MATS["olhar"] = S.node_mat("Olhar", (1.0, 0.85, 0.15), emit=8.0)  # sem uso no pov.blend, não foi salvo
pov_cam = bpy.data.objects["Olhar do rapaz"]
pov_cam.data.sensor_fit = "VERTICAL"
pov_cam.data.angle = math.radians(EYE_FOV)
b = S.Builder()
S.mannequin(b, *HIM)
him = b.build("Rapaz", Matrix.Identity(4))


def at(frame):
    """Põe a cena no quadro do vídeo e o rapaz onde a câmera dele está, virado para onde ela olha."""
    sc.frame_set(frame - F0 + 1)
    i = info[frame - F0]
    eye, tgt = Vector(i["eye"]), Vector(i["target"])
    d = (tgt - eye).to_2d()
    him.matrix_world = Matrix.Translation((eye.x, eye.y, 0)) @ Matrix.Rotation(math.atan2(d.y, d.x), 4, "Z")
    return pov_cam.matrix_world.copy()


def camera(name, fov, fit="VERTICAL"):
    """Câmera com campo de visão fov (graus): vertical, porque o quadro é em pé."""
    c = bpy.data.cameras.new(name)
    c.sensor_fit = fit
    c.angle = math.radians(fov)
    c.clip_start, c.clip_end = 0.05, 100
    o = bpy.data.objects.new(name, c)
    sc.collection.objects.link(o)
    return o


def spline(keys, u):
    """Catmull-Rom pelos valores de keys (lista de Vector), u de 0 a 1 entre o primeiro e o último."""
    n = len(keys) - 1
    x = min(u * n, n - 1e-9)
    i = int(x)
    t = x - i
    p0, p1, p2, p3 = keys[max(i - 1, 0)], keys[i], keys[i + 1], keys[min(i + 2, n)]
    return 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)


def ease(u):
    return u * u * u * (u * (6 * u - 15) + 10)


def render(path):
    if not path.exists():
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)


def fly(name, cam, poses, n, before=None, only=None):
    """Renderiza n quadros (ou só os de only) com a câmera passando pelas poses
    (posição, direção, campo de visão, para cima, deslocamento da lente). As poses são
    igualmente espaçadas no caminho; o tempo acelera e freia nas pontas."""
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    keys = [(Vector(p), Vector(f).normalized(), Vector((a, s, 0)), Vector(t)) for p, f, a, t, s in poses]
    for k in (only if only is not None else range(n)):
        u = ease(k / (n - 1))
        p, f, (a, s, _), t = (spline([key[j] for key in keys], u) for j in range(4))
        f = f.normalized()
        if only is None and k % args.every:
            if before:  # mantém o registro de cada quadro, mesmo os pulados na prévia
                before(p)
            continue
        x = f.cross(t).normalized()
        cam.matrix_world = Matrix.Translation(p) @ Matrix((x, -f.cross(x), -f)).transposed().to_4x4()
        cam.data.angle, cam.data.shift_x = math.radians(a), s
        if before:
            before(p)
        render(folder / f"{k + 1:04d}.png")


def look_from(p, target, fov):
    return (p, Vector(target) - Vector(p), fov, (0, 0, 1), 0.0)


# ---------------------------------------------------------------- render

if args.preview:
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "MATERIAL"
    sc.render.resolution_x, sc.render.resolution_y = W * 2 // 5, H * 2 // 5
else:  # a GPU fica nas preferências, não no .blend: sem isto, o Cycles cai na CPU (5x mais lento)
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True
    sc.cycles.device = "GPU"
    sc.cycles.samples = args.samples
    sc.render.resolution_x, sc.render.resolution_y = W, H
out.mkdir(parents=True, exist_ok=True)

# cctv: a mesma câmera do store.py, com sobra para a lente. 3200 px porque o promo.py
# recorta o meio em 4:5 e amplia 1,25x
if args.shot in (None, "cctv") and not (out / "cctv_raw.png").exists():
    at(FREEZE)
    cctv = camera("CCTV", math.degrees(2 * math.atan(calib.OVERSCAN)), "HORIZONTAL")
    Mc = Matrix(S.CAM["M"])
    right, down, fwd = Mc.col[0], Mc.col[1], Mc.col[2]
    cctv.matrix_world = Matrix((
        (right.x, -down.x, -fwd.x, 0), (right.y, -down.y, -fwd.y, 0), (right.z, -down.z, -fwd.z, S.CAM["h"]),
        (0, 0, 0, 1),
    ))
    w = 3200 if not args.preview else 825
    res = (sc.render.resolution_x, sc.render.resolution_y)
    sc.render.resolution_x, sc.render.resolution_y = w, round(w * 0.40 / calib.OVERSCAN)
    sc.camera = cctv
    render(out / "cctv_raw.png")
    sc.render.resolution_x, sc.render.resolution_y = res

# voo: da CCTV até os olhos dele, com o tempo parado em 18,5 s. A câmera fica sempre atrás
# dele (o fundo é a loja, não a vitrine); perto da cabeça, renderiza também sem o boneco,
# e o promo.py funde as duas versões para ele sumir aos poucos.
if args.shot in (None, "voo"):
    end = at(FREEZE)
    eye = end.translation.copy()
    head = Vector((eye.x, eye.y, 1.55))
    look = -end.col[2].xyz
    ahead = Vector((look.x, look.y, 0)).normalized()
    side = Vector((-ahead.y, ahead.x, 0))  # à esquerda dele
    up = Vector((0, 0, 1))
    vfov = math.degrees(2 * math.atan(calib.H / 2 / S.CAM["f"]))  # a altura toda da CCTV
    Mc = Matrix(S.CAM["M"])
    poses = [  # a câmera de segurança, inclinada como ela e com a lente deslocada para o recorte
        ((0, 0, S.CAM["h"]), Mc.col[2].xyz, vfov, -Mc.col[1].xyz, (CX - calib.W / 2) / calib.H),
        look_from(head - ahead * 1.9 + side * 0.9 + up * 0.95, head + ahead * 2.5, 50),
        look_from(head - ahead * 1.5 + side * 0.7 + up * 0.5, head + ahead * 2.2, 58),
        look_from(head - ahead * 0.9 + side * 0.1 + up * 0.15, head + ahead * 3.0 - up * 0.2, 64),
        (eye, look, EYE_FOV, up, 0.0),  # nos olhos: a câmera do pov.py
    ]
    cam = camera("Voo", vfov)
    sc.camera = cam
    near = {}

    def body(p):
        d = (p - head).length
        him.hide_render = d < 0.45  # dentro da cabeça, nunca
        near[len(near)] = round(d, 3)

    fly("voo", cam, poses, 165, body)
    # versão sem o boneco dos quadros perto da cabeça (de 0,45 a 1,2 m)
    him.hide_render = True
    fly("voo_sem", cam, poses, 165, only=[k for k, d in near.items() if 0.45 <= d < 1.2])
    him.hide_render = False
    (out / "voo.json").write_text(json.dumps({"head_distance": [near[k] for k in sorted(near)]}))

# visao: a câmera dele em pé, nos quadros do vídeo de origem que o promo.py mostra
if args.shot in (None, "visao"):
    him.hide_render = True
    sc.camera = pov_cam
    (out / "visao").mkdir(parents=True, exist_ok=True)
    frames = list(range(FREEZE, F_FAST + 1)) + list(range(F_FAST + 3, F1 + 1, 3))
    for f in frames[:: args.every]:
        at(f)
        render(out / "visao" / f"{f:04d}.png")
    him.hide_render = False

# mapa: sem teto, o caminho e os olhares dele até o fim do trecho
if args.shot in (None, "mapa"):
    end = at(F1)
    for o in bpy.data.objects:  # casa de bonecas: sem teto, vitrine e paredes da frente
        if o.name.startswith(("Loja fechada", "Placa da luminária", "Parede da direita", "Parede da entrada")):
            o.hide_render = True
    glow = S.node_mat("Caminho", (1.0, 0.55, 0.02), emit=1.2)  # mais forte, o AgX lava para branco
    S.MATS["caminho"] = glow
    trail = S.Builder()
    pts = [Vector((i["eye"][0], i["eye"][1], 0.012)) for i in info]
    for a, c in zip(pts[::3], pts[3::3]):  # fita no piso, em pedaços de 3 quadros
        d = c - a
        if d.length < 1e-4:
            continue
        nrm = Vector((-d.y, d.x, 0)).normalized() * 0.055
        trail._add([a - nrm, a + nrm, c + nrm, c - nrm], [(0, 1, 2, 3)], "caminho", None)
    for p in (pts[0], pts[-1]):
        trail.cyl("caminho", p.x, p.y, 0.005, 0.02, 0.12, n=24)
    trail.build("Caminho", Matrix.Identity(4))
    # pontos para onde ele olhou: o raio da CCTV pelo pico do Gaze-LLE até a loja (como no pov.py)
    data = json.loads((args.run_dir / "trajectories.json").read_text())["people"]
    people = [o for o in bpy.data.objects if o.name.startswith(("Pessoa", "Escada", "Rapaz"))]
    for o in people:  # o raio só bate na loja, como no pov.py
        o.hide_viewport = True
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dots = S.Builder()
    for r in data["12"]:
        if "gaze_ray" in r and F0 <= r["frame"] <= F1:
            hit, loc, *_ = sc.ray_cast(dg, S.eye_cam, Vector(r["gaze_ray"]), distance=40)
            if hit:
                dots.sphere("caminho", loc.x, loc.y, loc.z, 0.045, nu=10, nv=6)
    dots.build("Olhares", Matrix.Identity(4))
    for o in people:
        o.hide_viewport = False
    eye = end.translation.copy()
    tgt = Vector(info[-1]["target"])
    beam = S.Builder()
    beam.cyl("olhar", 0, 0, 0, (tgt - eye).length, 0.012, n=8)
    beam.sphere("olhar", 0, 0, (tgt - eye).length, 0.06)
    beam.build("Olhar", Matrix.Translation(eye) @ (tgt - eye).to_track_quat("Z", "Y").to_matrix().to_4x4())
    cam = camera("Mapa", 55)
    sc.camera = cam
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25  # sem teto, a luz de fora sobra
    poses = [
        look_from((-2.5, -5.5, 10.5), (2.6, 2.6, 0.3), 58),
        look_from((0.0, -6.0, 8.0), (2.8, 2.9, 0.4), 56),
        look_from((2.2, -5.0, 6.2), (3.1, 3.1, 0.5), 54),
    ]
    fly("mapa", cam, poses, 225)
