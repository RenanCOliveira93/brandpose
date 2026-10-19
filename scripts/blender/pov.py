"""Câmera subjetiva: o que o rapaz de jaqueta azul via, a partir de 17,3 s do vídeo.

No rastreador ele é a pessoa 12 (depois 18 e 21; trajectories.py junta os pedaços).

Posição: os pés de trajectories.py, suavizados, com os olhos a 1,62 m, a meio metro
de móveis e paredes, e um leve balanço ao andar. Direção: o olhar do Gaze-LLE (a
cada 3 quadros), levado à cena pelo raio da câmera CCTV; onde falta olhar, vale a
direção da caminhada. As outras pessoas andam pelos próprios trajetos. Para a
visão de dentro, a loja ganha teto, vitrine na frente e as paredes que a CCTV não
vê (inventadas).

Uso: Blender -b --python pov.py -- RUN_DIR [--range F0 F1] [--samples N] [--lang en] [--every N] [--frame F]
  --range: quadros do vídeo original (padrão: 519 a 1050, de 17,3 s a 35 s; ele fica até 1809, o fim).
  --frame F: só o quadro F do vídeo original, para teste.
Grava em RUN_DIR/pov_<idioma>/: frames/, pov_info.json (olho, alvo e zona por quadro) e pov.blend.
O render é retomável: rodando de novo, pula os quadros já gravados.
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

import store as S

ap = argparse.ArgumentParser(prog="pov.py")
ap.add_argument("run_dir", type=Path, help="pasta da rodada, com o trajectories.json")
ap.add_argument("--range", type=int, nargs=2, default=[519, 1050], metavar=("F0", "F1"),
                help="quadros do vídeo original")
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--lang", choices=["pt", "en"], default="pt", help="idioma das placas")
ap.add_argument("--every", type=int, default=1, help="renderiza 1 a cada N quadros (prévia)")
ap.add_argument("--frame", type=int, help="só este quadro do vídeo original, para teste")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])

WHO = "12"
F0, F1 = args.range
N = F1 - F0 + 1
FPS = 30
EYE = 1.62
LENS = 70  # campo de visão horizontal, em graus
TAU = 30  # quem sai do quadro segue andando e para em ~1 s
GAP = 30  # sumiço maior que isso (quadros): sai de cena e volta onde reaparecer

# ---------------------------------------------------------------- séries no tempo


def gauss(a, sigma):
    """Média gaussiana ao longo do tempo (eixo 0), repetindo as bordas."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, [(r, r), (0, 0)], mode="edge")
    return np.stack([np.convolve(p[:, j], k, "valid") for j in range(a.shape[1])], 1)


def median(a, w):
    r = w // 2
    p = np.pad(a, [(r, r), (0, 0)], mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, w, axis=0), axis=-1)


def path(recs, stay=False):
    """Pés por quadro da janela; NaN quando a pessoa está fora de cena.

    Ao sumir da CCTV, a pessoa segue andando por ~1,5 s e sai de cena (com
    stay, para e fica). Sumiços curtos (até GAP quadros) são interpolados."""
    fr = np.array([r["frame"] for r in recs])
    xy = np.array([r["feet"] for r in recs], float)
    cuts = np.flatnonzero(np.diff(fr) > GAP) + 1
    pos = np.full((N, 2), np.nan)
    for seg in np.split(np.arange(len(fr)), cuts):
        f, p = fr[seg], xy[seg]
        full = np.arange(f[0], f[-1] + 1)
        s = np.stack([np.interp(full, f, p[:, k]) for k in range(2)], 1)
        s = gauss(median(s, 9), 4)
        keep = (full >= F0) & (full <= F1)
        pos[full[keep] - F0] = s[keep]
        last = seg[-1] == len(fr) - 1
        span = F1 - f[-1] if last and stay else min(45, F1 - f[-1])
        if span > 0:
            v = (s[-5] - s[-15]) / 10 if len(s) > 15 else np.zeros(2)
            t = np.arange(1, span + 1)
            ext = s[-1] + np.outer(TAU * (1 - np.exp(-t / TAU)), v)
            idx = f[-1] + t - F0
            ok = (idx >= 0) & (idx < N)
            pos[idx[ok]] = ext[ok]
    return pos


def direction(obs, sigma, fallback):
    """Regressão por núcleo gaussiano de vetores unitários (quadro, vetor, peso),
    com uma segunda passada sem as observações a mais de 50° da primeira."""
    if not obs:
        return np.tile(fallback, (N, 1))
    f = np.array([o[0] for o in obs], float)
    v = np.array([o[1] for o in obs], float)
    w = np.array([o[2] for o in obs], float)
    t = np.arange(N)[:, None]
    for _ in range(2):
        # núcleo largo e fraco: sem observações por perto, vale a média dos ~4 s vizinhos
        k = w * (np.exp(-0.5 * ((t - f) / sigma) ** 2) + 0.02 * np.exp(-0.5 * ((t - f) / (8 * sigma)) ** 2))
        d = k @ v
        n = np.linalg.norm(d, axis=1)
        prev = np.asarray(fallback, float)
        for i in range(N):  # sem nada mesmo: mantém a direção anterior
            d[i] = d[i] / n[i] if n[i] > 1e-9 else prev
            prev = d[i]
        cos = np.sum(d[f.astype(int)] * v, axis=1)
        w = np.where(cos < math.cos(math.radians(50)), 0.0, w)
    return d


# ---------------------------------------------------------------- loja fechada (só para a visão de dentro)

data = json.loads((args.run_dir / "trajectories.json").read_text())["people"]
fit = S.build(args.lang)
sc = bpy.context.scene
S.MATS["teto"] = S.node_mat("Teto", (0.86, 0.86, 0.84), rough=0.9)
closure = S.Builder()
closure.box("teto", -1.0, -3.8, 3.0, 16.0, 16.0, 3.05)
# vitrine na frente (atrás da CCTV): rodapé, vidro, faixa de cima e montantes
closure.box("parede", -0.92, -3.6, 0.0, -0.80, 7.6, 0.5)
closure.box("vidro", -0.88, -3.6, 0.5, -0.84, 7.6, 2.5)
closure.box("parede", -0.92, -3.6, 2.5, -0.80, 7.6, 3.0)
for y in np.arange(-3.6, 7.7, 1.6):
    closure.box("preto", -0.93, y - 0.03, 0.5, -0.79, y + 0.03, 2.5)
closure.build("Loja fechada", Matrix.Identity(4))
cM, cL, cd, _ = fit["CERVEJAS"]
S.wall("Parede da direita", (-0.92, -3.6), (cM.translation.x + cd + 0.02, -3.6), "parede")
oM, _, od, _ = fit["ÓCULOS"]
o_start = (oM.translation + oM.col[1].xyz * (od + 0.02)).to_2d()
S.wall("Parede da entrada", (-0.92, 7.6), tuple(o_start), "parede")
# na foto, a borda esquerda mostra prateleiras antes do rack de óculos
a = (o_start - Vector((-0.92, 7.6))).normalized()
n = Vector((-a.y, a.x))
p0 = o_start - a * 3.7 - n * 0.47
g = S.Builder()
S.gondola(g, 3.5, 0.45, 1.6, "box", sides=1)
g.build("Prateleira da entrada", Matrix(((a.x, n.x, 0, p0.x), (a.y, n.y, 0, p0.y), (0, 0, 1, 0), (0, 0, 0, 1))))

# ---------------------------------------------------------------- olhares na cena

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()


def cast(origin, d):
    hit, loc, _, _, obj, _ = sc.ray_cast(depsgraph, Vector(origin), Vector(d), distance=40)
    return (np.array(loc), obj.name) if hit else (None, None)


for recs in data.values():
    for r in recs:
        if "gaze_ray" in r and F0 - 30 <= r["frame"] <= F1:
            r["hit"] = cast(S.eye_cam, r["gaze_ray"])[0]


def clear_of_fixtures(pos, radius=0.5):
    """Mantém o olho a meio metro de móveis e paredes: o trajeto erra uns
    20-30 cm e a câmera não pode entrar num rack."""
    dirs = [Vector((math.cos(a), math.sin(a), 0)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    out = pos.copy()
    for i in range(N):
        for _ in range(6):
            push = []
            for z in (EYE, 1.0):
                o = Vector((out[i, 0], out[i, 1], z))
                for d in dirs:
                    hit, loc, *_ = sc.ray_cast(depsgraph, o, d, distance=radius)
                    if hit:
                        push.append(-np.array(d[:2]) * (radius - (loc - o).length))
            if not push:
                break
            out[i] += 1.5 * np.mean(push, 0)
    return gauss(out, 4)


# ---------------------------------------------------------------- a câmera dele

recs = [r for r in data[WHO] if F0 <= r["frame"] <= F1]
pos = clear_of_fixtures(path(data[WHO], stay=True))
vel = np.gradient(pos, axis=0) * FPS
speed = np.linalg.norm(vel, axis=1)
eye = np.column_stack([pos, np.full(N, EYE)])
obs = []
for r in recs:
    if r.get("hit") is not None:
        i = r["frame"] - F0
        d = r["hit"] - eye[i]
        d /= np.linalg.norm(d)
        pitch = np.clip(math.asin(d[2]), math.radians(-30), math.radians(15))
        h = d[:2] / np.linalg.norm(d[:2]) * math.cos(pitch)
        obs.append((i, [h[0], h[1], math.sin(pitch)], 1.0))
for i in range(0, N, 3):  # andando, olha um pouco para a frente
    if speed[i] > 0.4:
        h = vel[i] / speed[i] * math.cos(math.radians(10))
        obs.append((i, [h[0], h[1], -math.sin(math.radians(10))], 0.35))
look_dir = direction(obs, 7, np.array([1.0, 0, 0]))

walked = np.cumsum(speed / FPS)
bob = 0.015 * np.minimum(speed / 1.2, 1) * np.sin(2 * math.pi * walked / 0.7)
yaw = np.unwrap(np.arctan2(look_dir[:, 1], look_dir[:, 0]))
pitch = np.arcsin(np.clip(look_dir[:, 2], -1, 1))

cam = S.look("Olhar do rapaz", (0, 0, 0), (1, 0, 0), lens_deg=LENS)
cam.data.clip_start = 0.05
info = []
for i in range(N):
    cam.location = (pos[i, 0], pos[i, 1], EYE + bob[i])
    cam.rotation_euler = (math.pi / 2 + pitch[i], 0, yaw[i] - math.pi / 2)
    cam.keyframe_insert("location", frame=i + 1)
    cam.keyframe_insert("rotation_euler", frame=i + 1)
    hit, name = cast(cam.location, look_dir[i])
    target = hit if hit is not None else eye[i] + 5 * look_dir[i]
    info.append({"frame": F0 + i, "eye": [round(float(v), 3) for v in eye[i]],
                 "target": [round(float(v), 3) for v in target], "zone": S.ZONE_OF.get(name)})

# ---------------------------------------------------------------- as outras pessoas

for pid in S.LOOKS:
    obj = S.person(pid).build(f"Pessoa {pid}", Matrix.Identity(4))
    own = [r for r in data[pid] if F0 - 30 <= r["frame"] <= F1]
    if pid in S.LADDER:
        x, y, z0 = S.LADDER[pid]
        p = np.tile([x, y], (N, 1))
    else:
        z0 = 0.0
        p = path(data[pid])
    v = np.gradient(np.nan_to_num(p), axis=0) * FPS
    sp = np.linalg.norm(v, axis=1)
    ob = []
    for r in own:
        i = r["frame"] - F0
        if r.get("hit") is not None and 0 <= i < N and not np.isnan(p[i, 0]):
            d = r["hit"][:2] - p[i]
            ob.append((i, d / np.linalg.norm(d), 0.6))
    for i in range(N):
        if sp[i] > 0.3 and not np.isnan(p[i, 0]):
            ob.append((i, v[i] / sp[i], 1.0))
    face = direction(ob, 8, np.array([-1.0, 0]))
    fyaw = np.unwrap(np.arctan2(face[:, 1], face[:, 0]))
    if pid in S.LADDER:  # escada na frente dele, virada para onde ele trabalha
        S.ladder(z0).build("Escada", Matrix.Translation((x, y, 0)) @ Matrix.Rotation(float(np.median(fyaw)), 4, "Z"))
    for i in range(N):
        visible = not np.isnan(p[i, 0])
        obj.hide_render = not visible
        obj.keyframe_insert("hide_render", frame=i + 1)
        if visible:
            obj.location = (p[i, 0], p[i, 1], z0)
            obj.rotation_euler = (0, 0, fyaw[i])
            obj.keyframe_insert("location", frame=i + 1)
            obj.keyframe_insert("rotation_euler", frame=i + 1)

# ---------------------------------------------------------------- render

S.use_cycles(args.samples)
sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6  # luz do dia pela vitrine
S.add_lights(z=2.94, energy=55, panels=True)
sc.render.use_persistent_data = True
cy = sc.cycles  # menos rebotes e amostragem adaptativa: ~2x mais rápido, sem diferença visível
cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces = 6, 2, 2
cy.transmission_bounces, cy.transparent_max_bounces = 6, 8
cy.caustics_reflective = cy.caustics_refractive = False
cy.adaptive_threshold = 0.03
cy.denoising_use_gpu = True
S.MATS["luz"].cycles.emission_sampling = "NONE"  # quem ilumina são as luzes de área
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.render.fps = FPS
sc.camera = cam
sc.frame_start, sc.frame_end, sc.frame_step = 1, N, args.every
out = args.run_dir / f"pov_{args.lang}"
out.mkdir(parents=True, exist_ok=True)
(out / "pov_info.json").write_text(json.dumps(info, ensure_ascii=False))
bpy.ops.wm.save_as_mainfile(filepath=str(out / "pov.blend"))
if args.frame is not None:
    sc.frame_set(args.frame - F0 + 1)
    sc.render.filepath = str(out / f"teste_{args.frame}.png")
    bpy.ops.render.render(write_still=True)
else:
    sc.render.filepath = str(out / "frames" / "####")
    sc.render.use_overwrite = False  # retomável: pula os quadros já gravados
    bpy.ops.render.render(animation=True)
