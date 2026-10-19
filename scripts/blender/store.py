"""A loja modelada à mão sobre a câmera calibrada, no Blender (Cycles).

O mundo é o sistema da loja (layout.py): x ao longo das gôndolas, y atravessando,
z para cima, câmera de segurança sobre a origem. Móveis e câmera vêm de calib.py
(cantos marcados na foto). Cada zona ganha uma placa com o nome e a atenção
estimada, e um brilho no piso com a cor do calor. Produtos, paredes e luzes são
genéricos (inventados).

Rodando, põe na cena as pessoas de um quadro (trajectories.py), cada uma com o
olhar até onde o raio da câmera pelo pico do Gaze-LLE bate, e renderiza quatro
vistas: a da CCTV (sem distorção; warp.py aplica a lente), órbita, corredor e
planta. Importado (pov.py), só monta a loja, com build().

Uso: Blender -b --python store.py -- RUN_DIR [--frame 279] [--lang en] [--samples 160] [--preview]
Grava em RUN_DIR/loja_<idioma>/: cctv_raw.png, orbita.png, corredor.png, planta.png e loja.blend.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # o Blender não põe a pasta do script no caminho

import bpy
import numpy as np
from mathutils import Matrix, Vector

import calib
import layout

CAM = calib.load_camera()
eye_cam = Vector((0, 0, CAM["h"]))  # a lente da CCTV: os raios do olhar saem daqui
rng = random.Random(7)

DRINK_COLORS = [
    (0.20, 0.06, 0.03), (0.85, 0.35, 0.05), (0.15, 0.55, 0.15), (0.75, 0.85, 0.95), (0.85, 0.10, 0.08),
    (0.95, 0.80, 0.15), (0.10, 0.20, 0.60), (0.55, 0.75, 0.25),
]
LABEL_COLORS = [(0.9, 0.9, 0.88), (0.8, 0.1, 0.1), (0.1, 0.1, 0.12), (0.95, 0.75, 0.1), (0.1, 0.3, 0.7)]
PRODUCT_COLORS = [
    (0.75, 0.08, 0.06), (0.95, 0.45, 0.05), (0.95, 0.80, 0.10), (0.10, 0.25, 0.70), (0.85, 0.85, 0.82),
    (0.35, 0.18, 0.08), (0.10, 0.50, 0.20), (0.45, 0.10, 0.45), (0.05, 0.05, 0.06), (0.60, 0.80, 0.95),
]
SHIRTS = [(0.55, 0.10, 0.10), (0.15, 0.30, 0.55), (0.20, 0.45, 0.25), (0.85, 0.85, 0.85), (0.60, 0.45, 0.15)]
PANTS = [(0.10, 0.14, 0.25), (0.15, 0.15, 0.16), (0.35, 0.30, 0.22), (0.12, 0.20, 0.35), (0.05, 0.05, 0.06)]
HAIRS = [(0.05, 0.03, 0.02), (0.25, 0.15, 0.07), (0.02, 0.02, 0.02), (0.55, 0.40, 0.20), (0.12, 0.08, 0.05)]
SKINS = [(0.80, 0.60, 0.45), (0.55, 0.38, 0.26), (0.92, 0.74, 0.60), (0.40, 0.26, 0.18), (0.75, 0.55, 0.40)]
# quem aparece na câmera subjetiva, com a roupa do vídeo: camisa, pele, calça e cabelo
# (careca: cabelo da cor da pele). Os demais usam as paletas acima.
LOOKS = {
    "8": ((0.04, 0.04, 0.05), (0.80, 0.60, 0.45), (0.05, 0.05, 0.06), (0.03, 0.02, 0.02)),
    "14": ((0.70, 0.80, 0.92), (0.85, 0.66, 0.52), (0.10, 0.16, 0.30), (0.85, 0.66, 0.52)),
    "15": ((0.55, 0.56, 0.55), (0.35, 0.22, 0.15), (0.28, 0.27, 0.18), (0.35, 0.22, 0.15)),
    "4": ((0.06, 0.06, 0.08), (0.75, 0.55, 0.40), (0.08, 0.08, 0.10), (0.05, 0.03, 0.02)),
}
# o repositor fica numa escada: posição e altura do degrau ajustadas para nariz e tornozelos baterem
LADDER = {"4": (7.96, 2.10, 0.69)}

MATS = {}
ZONE_OF = {}  # nome do objeto -> zona ("short" do zones_acti.json): o que o olhar atinge

# ---------------------------------------------------------------- geometria


class Builder:
    """Junta caixas, cilindros e esferas num só objeto, com um material por peça."""

    def __init__(self):
        self.verts, self.faces, self.mats, self.cols = [], [], [], []
        self.mat_names = []

    def _mat(self, name):
        if name not in self.mat_names:
            self.mat_names.append(name)
        return self.mat_names.index(name)

    def _add(self, verts, faces, mat, color):
        base = len(self.verts)
        self.verts += verts
        mi = self._mat(mat)
        for f in faces:
            self.faces.append([base + i for i in f])
            self.mats.append(mi)
            self.cols.append((*(color or (1, 1, 1)), 1))

    def box(self, mat, x0, y0, z0, x1, y1, z1, color=None):
        v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        self._add(v, f, mat, color)

    def cyl(self, mat, cx, cy, z0, z1, r, n=10, color=None, ry=None, top=1.0):
        """Cilindro (elíptico com ry; top < 1 afina em cima)."""
        ry = ry or r
        ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
        v = [(cx + r * k * c, cy + ry * k * s, z) for z, k in ((z0, 1.0), (z1, top)) for c, s in ring]
        f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        f += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        self._add(v, f, mat, color)

    def sphere(self, mat, cx, cy, cz, r, color=None, nu=12, nv=8):
        v = [(cx, cy, cz - r)]
        for j in range(1, nv):
            phi = math.pi * j / nv - math.pi / 2
            for i in range(nu):
                th = 2 * math.pi * i / nu
                v.append((cx + r * math.cos(phi) * math.cos(th), cy + r * math.cos(phi) * math.sin(th),
                          cz + r * math.sin(phi)))
        v.append((cx, cy, cz + r))
        top = len(v) - 1
        ring = lambda j, i: 1 + (j - 1) * nu + i % nu
        f = [(0, ring(1, i + 1), ring(1, i)) for i in range(nu)]
        for j in range(1, nv - 1):
            f += [(ring(j, i), ring(j, i + 1), ring(j + 1, i + 1), ring(j + 1, i)) for i in range(nu)]
        f += [(ring(nv - 1, i), ring(nv - 1, i + 1), top) for i in range(nu)]
        self._add(v, f, mat, color)

    def build(self, name, matrix):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.verts, [], self.faces)
        me.polygons.foreach_set("material_index", self.mats)
        attr = me.color_attributes.new("col", "FLOAT_COLOR", "FACE")
        attr.data.foreach_set("color", [c for col in self.cols for c in col])
        for m in self.mat_names:
            me.materials.append(MATS[m])
        obj = bpy.data.objects.new(name, me)
        obj.matrix_world = matrix
        bpy.context.scene.collection.objects.link(obj)
        return obj


def fill_shelf(b, x0, x1, y0, y1, z, gap, style):
    """Produtos lado a lado numa prateleira (face da prateleira em y0)."""
    x = x0 + 0.02
    while x < x1 - 0.05:
        w = rng.uniform(0.06, 0.16) if style != "can" else 0.075
        if x + w > x1 - 0.02:
            break
        h = min(gap - 0.05, rng.uniform(0.12, 0.30) if style == "bag" else rng.uniform(0.10, 0.22))
        col = rng.choice(PRODUCT_COLORS)
        rows = 3
        for k in range(rows):  # fileiras de trás repetem a da frente
            yy = y0 + 0.03 + k * (y1 - y0 - 0.05) / rows
            dd = (y1 - y0 - 0.05) / rows - 0.01
            if style == "can":
                b.cyl("produto", x + w / 2, yy + dd / 2, z, z + min(h, 0.14), w / 2 - 0.004, n=8, color=col)
            else:
                b.box("produto", x, yy, z, x + w - 0.008, yy + dd, z + h, color=col)
        x += w


def gondola(b, L, d, H, style, sides=2):
    b.box("base", 0, 0.03, 0, L, d - 0.03, 0.12)
    mid = d / 2 if sides == 2 else d - 0.03
    b.box("metal", 0.02, mid - 0.02, 0.12, L - 0.02, mid + 0.02, H)
    for x in (0, L - 0.03):
        b.box("metal", x, 0, 0, x + 0.03, d, H)
    levels = np.linspace(0.12, H - 0.30, 5 if H >= 1.5 else 4)
    gap = levels[1] - levels[0]
    faces = [(0.0, mid - 0.02)] + ([(d, mid + 0.02)] if sides == 2 else [])
    for front, back in faces:
        lo, hi = min(front, back), max(front, back)
        for z in levels:
            b.box("metal", 0.03, lo, z, L - 0.03, hi, z + 0.02)
            rail = (front - 0.015, front) if front < back else (front, front + 0.015)
            b.box("trilho", 0.03, rail[0], z - 0.02, L - 0.03, rail[1], z + 0.04)
            if front < back:
                fill_shelf(b, 0.03, L - 0.03, lo, hi, z + 0.02, gap, style)
            else:  # lado de trás: espelha a ocupação
                n0 = len(b.verts)
                fill_shelf(b, 0.03, L - 0.03, lo, hi, z + 0.02, gap, style)
                for i in range(n0, len(b.verts)):
                    x, y, zz = b.verts[i]
                    b.verts[i] = (x, lo + hi - y, zz)
                nf = len(b.faces)  # espelhar inverte as faces
                for k in range(nf - 1, -1, -1):
                    if b.faces[k][0] >= n0:
                        b.faces[k] = b.faces[k][::-1]
                    else:
                        break


def fridge(b, L, d, H):
    b.box("preto", 0, 0, 0, L, d, 0.15)
    b.box("preto", 0, 0, H - 0.25, L, d, H)
    b.box("luz", 0.02, d - 0.06, 0.15, L - 0.02, d - 0.02, H - 0.25)
    doors = max(1, round(L / 0.76))
    w = L / doors
    for i in range(doors + 1):
        b.box("aco", i * w - 0.03, 0, 0.15, i * w + 0.03, 0.05, H - 0.25)
    for i in range(1, doors + 1):  # puxadores
        b.box("aco", i * w - 0.09, -0.03, 0.8, i * w - 0.07, 0.0, 1.5)
    for z in np.linspace(0.20, H - 0.55, 5):
        b.box("aco", 0.03, 0.10, z - 0.015, L - 0.03, d - 0.06, z)
        x = 0.06
        while x < L - 0.08:  # uma marca por coluna, em fila para o fundo
            col = rng.choice(DRINK_COLORS)
            hb = rng.uniform(0.15, 0.22)
            for k in range(4):
                y = 0.14 + k * 0.13
                b.cyl("garrafa", x, y, z, z + hb, 0.032, n=10, color=col)
                b.cyl("garrafa", x, y, z + hb, z + hb + 0.05, 0.032, n=10, color=col, top=0.4)
                b.cyl("tampa", x, y, z + hb + 0.05, z + hb + 0.07, 0.014, n=8)
                b.cyl("rotulo_col", x, y, z + 0.05, z + 0.11, 0.0335, n=10, color=rng.choice(LABEL_COLORS))
            x += 0.072
    b.box("vidro", 0.03, 0.03, 0.15, L - 0.03, 0.045, H - 0.25)


def wine(b, L, d, H):
    for x in (0, L - 0.04):
        b.box("madeira", x, 0, 0, x + 0.04, d, H)
    for z in np.linspace(0.05, H - 0.3, 5):
        b.box("madeira", 0.04, 0, z, L - 0.04, d, z + 0.03)
        for x in np.arange(0.09, L - 0.06, 0.09):
            col = rng.choice([(0.05, 0.18, 0.08), (0.30, 0.03, 0.05), (0.85, 0.80, 0.55)])
            b.cyl("garrafa", x, d / 2, z + 0.03, z + 0.33, 0.035, n=8, color=col)


def wallrack(b, L, d, H):
    b.box("preto", 0, d - 0.05, 0, L, d, H)
    b.box("base", 0, 0.05, 0, L, d - 0.05, 0.3)
    for z in np.arange(0.45, H - 0.1, 0.18):
        for x in np.arange(0.10, L - 0.08, 0.16):
            col = rng.choice([(0.05, 0.05, 0.05), (0.40, 0.25, 0.10), (0.60, 0.10, 0.10), (0.15, 0.15, 0.40)])
            b.box("produto", x - 0.06, d - 0.12, z, x + 0.06, d - 0.05, z + 0.05, color=col)


def counter(b, L, d, H):
    b.box("madeira", 0, 0.05, 0, L, d, H - 0.04)
    b.box("tampo", -0.02, 0, H - 0.04, L + 0.02, d, H)
    for i, x in enumerate(np.linspace(0.3, L - 0.7, 3)):
        b.box("preto", x, d - 0.45, H, x + 0.4, d - 0.05, H + 0.65)
        b.box("tela", x + 0.08, d - 0.452, H + 0.35, x + 0.32, d - 0.45, H + 0.55)
    for x in np.arange(L - 0.35, L - 0.05, 0.1):
        b.cyl("tampo", x, 0.25, H, H + 0.35, 0.04, n=8)


def display(b, L, d, H):
    b.box("madeira", 0, 0, 0, 0.03, d, H)
    b.box("madeira", L - 0.03, 0, 0, L, d, H)
    b.box("madeira", 0, d - 0.03, 0, L, d, H)
    for z in (0.05, 0.4, 0.75):
        b.box("madeira", 0.03, 0, z, L - 0.03, d - 0.03, z + 0.03)
        b.box("produto", 0.1, 0.08, z + 0.03, L - 0.1, d - 0.08, z + 0.2, color=(0.92, 0.92, 0.9))


def stand(b, L, d, H):
    """Expositor preto de chão (incensos e acessórios)."""
    b.box("preto", 0, 0, 0, L, d, 0.1)
    b.box("preto", 0, d - 0.03, 0, L, d, H)
    for x in (0, L - 0.025):
        b.box("preto", x, 0, 0, x + 0.025, d, H)
    for z in np.linspace(0.12, H - 0.25, 3):
        b.box("preto", 0.025, 0.02, z, L - 0.025, d - 0.03, z + 0.015)
        fill_shelf(b, 0.025, L - 0.025, 0.02, d - 0.03, z + 0.015, 0.24, "bag")


def water(b, L, d, H):
    """Fardos de água empilhados."""
    for z in np.arange(0, H - 0.05, 0.24):
        for x in np.arange(0, L - 0.2, 0.34):
            for y in np.arange(0, d - 0.15, 0.27):
                b.box("fardo", x + 0.01, y + 0.01, z, x + 0.33, y + 0.26, z + 0.23)
                b.box("rotulo", x + 0.01, y + 0.005, z + 0.08, x + 0.33, y + 0.012, z + 0.13)


def freezer(b, L, d, H):
    """Freezer horizontal de sorvete."""
    b.box("azul", 0, 0, 0, L, d, H - 0.05)
    b.box("tampo", -0.01, -0.01, H - 0.05, L + 0.01, d + 0.01, H)
    b.box("vidro", 0.05, 0.05, H, L - 0.05, d - 0.05, H + 0.01)
    for x in np.arange(0.08, L - 0.1, 0.14):
        for y in np.arange(0.08, d - 0.1, 0.16):
            b.box("produto", x, y, H - 0.14, x + 0.12, y + 0.14, H - 0.06, color=rng.choice(PRODUCT_COLORS))


def mannequin(b, color, skin, pants, hair):
    """Boneco de proporções humanas (1,70 m), olhando para +X."""
    for y in (-0.095, 0.095):
        b.cyl("roupa", 0, y, 0.06, 0.50, 0.055, n=12, color=pants, top=1.25)  # canela
        b.sphere("roupa", 0, y, 0.50, 0.068, color=pants)  # joelho
        b.cyl("roupa", 0, y, 0.50, 0.92, 0.068, n=12, color=pants, top=1.35)  # coxa
        b.cyl("sapato", 0.04, y, 0.0, 0.07, 0.11, ry=0.05, n=12)
    b.cyl("roupa", 0, 0, 0.88, 1.02, 0.12, ry=0.19, n=16, color=pants)  # quadril
    b.cyl("roupa", 0, 0, 1.00, 1.40, 0.115, ry=0.175, n=16, color=color, top=1.15)  # tronco
    b.sphere("roupa", 0, 0, 1.40, 0.13, color=color, nu=16, nv=6)  # ombros
    for y in (-0.235, 0.235):
        b.sphere("roupa", 0, y, 1.38, 0.06, color=color)
        b.cyl("roupa", 0, y, 1.10, 1.38, 0.052, n=10, color=color, top=1.1)  # braço
        b.cyl("pele", 0.01, y, 0.84, 1.10, 0.042, n=10, color=skin, top=1.2)  # antebraço
        b.sphere("pele", 0.015, y, 0.80, 0.048, color=skin)  # mão
    b.cyl("pele", 0, 0, 1.43, 1.52, 0.048, n=10, color=skin)  # pescoço
    b.sphere("pele", 0.005, 0, 1.60, 0.105, color=skin, nu=16, nv=12)  # cabeça
    b.sphere("cabelo", -0.02, 0, 1.635, 0.104, color=hair, nu=16, nv=12)  # cabelo
    b.sphere("pele", 0.095, 0, 1.585, 0.025, color=skin)  # nariz


def person(pid, k=0):
    """Boneco da pessoa pid: a roupa de LOOKS ou a k-ésima das paletas genéricas."""
    b = Builder()
    mannequin(b, *LOOKS.get(pid, (SHIRTS[k % 5], SKINS[k % 5], PANTS[k % 5], HAIRS[k % 5])))
    return b


def ladder(z0):
    """Escada em A, de alumínio, com o degrau de cima na altura z0 (a pessoa fica em x = 0)."""
    b = Builder()
    for side in (-0.22, 0.22):
        for dx in (-0.30, 0.35):
            b.box("aco", dx - 0.025, side - 0.02, 0, dx + 0.025, side + 0.02, 1.27)
    for zz in np.arange(0.23, z0 + 0.01, 0.23):
        b.box("aco", -0.30, -0.22, zz - 0.03, 0.12, 0.22, zz)
    b.box("aco", -0.30, -0.24, 1.22, 0.35, 0.24, 1.27)
    return b


# ---------------------------------------------------------------- materiais


def material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)  # cor que o Workbench usa em MATERIAL
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*rgb, 1)
    return mat


def node_mat(name, color, rough=0.5, metal=0.0, emit=0.0, attr=False, transmission=0.0):
    mat = material(name, color)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if transmission:
        bsdf.inputs["Transmission Weight"].default_value = transmission
        bsdf.inputs["IOR"].default_value = 1.45
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*color, 1)
        bsdf.inputs["Emission Strength"].default_value = emit
    if attr:
        a = mat.node_tree.nodes.new("ShaderNodeAttribute")
        a.attribute_name = "col"
        mat.node_tree.links.new(a.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def tile_mat():
    mat = material("Piso", (0.78, 0.72, 0.62))
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.35
    tc = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0
    brick.inputs["Scale"].default_value = 1
    brick.inputs["Brick Width"].default_value = 0.45
    brick.inputs["Row Height"].default_value = 0.45
    brick.inputs["Mortar Size"].default_value = 0.006
    brick.inputs["Color1"].default_value = (0.80, 0.74, 0.64, 1)
    brick.inputs["Color2"].default_value = (0.74, 0.69, 0.60, 1)
    brick.inputs["Mortar"].default_value = (0.45, 0.42, 0.38, 1)
    nt.links.new(tc.outputs["Object"], brick.inputs["Vector"])
    nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def heat_patch_mat(name, color, strength):
    """Brilho no piso que some ao se afastar da face (eixo Y gerado = 0 na face)."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 0.6)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 1.0, 0.0
    ramp.inputs["To Max"].default_value = 0.55
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Y"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def heat(t):
    """0 → azul frio, 0,5 → amarelo, 1 → vermelho."""
    stops = [(0.0, (0.12, 0.30, 0.80)), (0.5, (1.0, 0.78, 0.10)), (1.0, (0.92, 0.12, 0.08))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return tuple(a + (b - a) * k for a, b in zip(c0, c1))
    return stops[-1][1]


# ---------------------------------------------------------------- cena


def face_matrix(start, angle):
    """Matriz da face: X ao longo dela, Y para dentro do corpo, Z para cima."""
    p0, a, b = layout.face_frame(start, angle, 0)
    return Matrix(((a[0], b[0], 0, p0[0]), (a[1], b[1], 0, p0[1]), (0, 0, 1, 0), (0, 0, 0, 1)))


def sign(parent, zone, text, L, y, z, color):
    """Placa acesa sobre o móvel, larga o bastante para o texto."""
    cu = bpy.data.curves.new(f"Texto {zone}", "FONT")
    cu.body = text
    cu.size = 0.10
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    cu.materials.append(node_mat("Letra", (0.98, 0.98, 0.98), emit=1.0))
    txt = bpy.data.objects.new(f"Texto {zone}", cu)
    txt.matrix_world = parent.matrix_world @ Matrix.Translation((L / 2, y - 0.025, z + 0.13)) @ \
        Matrix.Rotation(math.radians(90), 4, "X")
    bpy.context.scene.collection.objects.link(txt)
    bpy.context.view_layer.update()  # mede o texto
    w = max(min(L * 0.8, 1.6), 0.7, txt.dimensions.x + 0.16)
    board = Builder()
    MATS["placa"] = node_mat(f"Placa {zone}", color, rough=0.4, emit=0.25)
    board.box("placa", (L - w) / 2, y - 0.02, z, (L + w) / 2, y + 0.02, z + 0.26)
    return board.build(f"Placa {zone}", parent.matrix_world), txt


BUILDERS = {
    "gondola": lambda b, L, d, H, z: gondola(b, L, d, H, {"MERCEARIA": "can", "SNACKS": "bag"}.get(z, "box"),
                                             sides=2 if d > 0.6 else 1),
    "shelf": lambda b, L, d, H, z: gondola(b, L, d, H, "can", sides=1),
    "stand": lambda b, L, d, H, z: stand(b, L, d, H),
    "fridge": lambda b, L, d, H, z: fridge(b, L, d, H),
    "wine": lambda b, L, d, H, z: wine(b, L, d, H),
    "wallrack": lambda b, L, d, H, z: wallrack(b, L, d, H),
    "counter": lambda b, L, d, H, z: counter(b, L, d, H),
    "display": lambda b, L, d, H, z: display(b, L, d, H),
    "water": lambda b, L, d, H, z: water(b, L, d, H),
    "freezer": lambda b, L, d, H, z: freezer(b, L, d, H),
}


def wall(name, p0, p1, mat, height=3.0, thick=0.12):
    a = Vector(p1) - Vector(p0)
    w = Builder()
    w.box(mat, 0, 0, 0, a.length, thick, height)
    a.normalize()
    w.build(name, Matrix(((a.x, -a.y, 0, p0[0]), (a.y, a.x, 0, p0[1]), (0, 0, 1, 0), (0, 0, 0, 1))))


def build(lang="pt"):
    """Monta a loja numa cena vazia, com as placas no idioma pedido.

    Devolve, por zona, a matriz da face, o comprimento, a profundidade e a altura do móvel."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.world = bpy.data.worlds.new("Fundo")
    sc.world.color = (0.03, 0.03, 0.035)
    rng.seed(7)
    ZONE_OF.clear()
    MATS.clear()
    MATS.update({
        "base": node_mat("Base", (0.08, 0.08, 0.09), rough=0.6),
        "metal": node_mat("Metal", (0.82, 0.82, 0.80), rough=0.35, metal=0.3),
        "trilho": node_mat("Trilho", (0.05, 0.12, 0.45), rough=0.4),
        "produto": node_mat("Produto", (1, 1, 1), rough=0.45, attr=True),
        "garrafa": node_mat("Garrafa", (1, 1, 1), rough=0.08, attr=True, transmission=0.35),
        "preto": node_mat("Preto", (0.03, 0.03, 0.035), rough=0.5),
        "aco": node_mat("Aço", (0.6, 0.6, 0.62), rough=0.25, metal=0.9),
        "luz": node_mat("Luz", (0.9, 0.95, 1.0), emit=4.0),
        "vidro": node_mat("Vidro", (0.9, 0.95, 1.0), rough=0.02, transmission=1.0),
        "madeira": node_mat("Madeira", (0.62, 0.45, 0.28), rough=0.6),
        "tampo": node_mat("Tampo", (0.88, 0.87, 0.84), rough=0.3),
        "tela": node_mat("Tela", (0.9, 0.6, 0.3), emit=3.0),
        "roupa": node_mat("Roupa", (1, 1, 1), rough=0.8, attr=True),
        "pele": node_mat("Pele", (1, 1, 1), rough=0.5, attr=True),
        "parede": node_mat("Parede", (0.80, 0.78, 0.72), rough=0.9),
        "parede_azul": node_mat("Parede azul", (0.04, 0.08, 0.35), rough=0.9),
        "fardo": node_mat("Fardo", (0.80, 0.88, 0.95), rough=0.15, transmission=0.6),
        "rotulo": node_mat("Rótulo", (0.70, 0.10, 0.12), rough=0.4),
        "azul": node_mat("Azul", (0.08, 0.25, 0.65), rough=0.4),
        "piso": tile_mat(),
        "tampa": node_mat("Tampa", (0.85, 0.85, 0.85), rough=0.3, metal=0.6),
        "rotulo_col": node_mat("Rótulo", (1, 1, 1), rough=0.5, attr=True),
        "sapato": node_mat("Sapato", (0.04, 0.04, 0.04), rough=0.4),
        "cabelo": node_mat("Cabelo", (1, 1, 1), rough=0.6, attr=True),
        "olhar": node_mat("Olhar", (1.0, 0.85, 0.15), emit=8.0),
    })
    MATS["vidro"].diffuse_color = (0.8, 0.9, 1.0, 0.15)

    names, objects = layout.zone_names(lang), layout.zone_names("pt")
    top = max(layout.ATTENTION.values())
    fit = {}
    for zone, kind, start, angle, L, d, H in layout.FIXTURES:
        b = Builder()
        BUILDERS[kind](b, L, d, H, zone)
        M = face_matrix(start, angle)
        obj = b.build(objects.get(zone, zone.strip("_").capitalize()), M)
        fit[zone] = (M, L, d, H)
        if zone not in names:
            continue
        att = layout.ATTENTION[zone]
        col = heat(att / top)
        sy = d / 2 if kind == "gondola" and d > 0.6 else -0.02
        parts = sign(obj, zone, f"{names[zone]}  {att:.0f}%", L, sy, H + (0.9 if kind == "counter" else 0.05), col)
        patch = bpy.data.meshes.new(f"Calor {zone}")
        patch.from_pydata([(0, -1.0, 0.004), (L, -1.0, 0.004), (L, 0, 0.004), (0, 0, 0.004)], [], [(0, 1, 2, 3)])
        patch.materials.append(heat_patch_mat(f"Calor {zone}", col, 0.3 + 1.7 * att / top))
        po = bpy.data.objects.new(f"Calor {zone}", patch)
        po.matrix_world = M
        sc.collection.objects.link(po)
        for o in (obj, po, *parts):
            ZONE_OF[o.name] = zone

    # paredes atrás dos móveis do fundo (sem a da frente e a da direita, como casa de bonecas)
    gM, _, gd, _ = fit["GELADEIRAS"]
    cM, cL, cd, _ = fit["CERVEJAS"]
    back_x = gM.translation.x + gd + 0.02
    beer_x = cM.translation.x + cd + 0.02
    corner_y = cM.translation.y + 0.05
    wall("Parede do fundo", (back_x, corner_y), (back_x, 10.6), "parede")
    wall("Parede das cervejas", (beer_x, corner_y - cL - 1.5), (beer_x, corner_y), "parede_azul")
    wall("Parede da quina", (beer_x, corner_y), (back_x, corner_y), "parede_azul")
    kM, kL, kd, _ = fit["CAFÉ"]
    cafe_x = kM.translation.x + kd + 0.5
    wall("Parede do café", (cafe_x, 10.6), (back_x + 0.12, 10.6), "parede")
    wall("Parede do café 2", (cafe_x, 10.6), (cafe_x, 15.0), "parede")
    oM, oL, od, _ = fit["ÓCULOS"]
    o0, oa = oM.translation, oM.col[0].xyz
    ob = oM.col[1].xyz * (od + 0.02)
    wall("Parede esquerda", (o0 + ob).to_2d(), (o0 + ob + oa * 9.5).to_2d(), "parede")

    floor = Builder()
    floor.box("piso", -2.0, -4.0, -0.02, 16.0, 16.0, 0)
    floor.build("Piso", Matrix.Identity(4))
    return fit


def look(name, pos, target, lens_deg=55, ortho=None):
    c = bpy.data.cameras.new(name)
    c.sensor_fit = "HORIZONTAL"
    c.angle = math.radians(lens_deg)
    c.clip_end = 100
    if ortho:
        c.type, c.ortho_scale = "ORTHO", ortho
    o = bpy.data.objects.new(name, c)
    o.location = pos
    o.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(o)
    return o


def use_cycles(samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Punchy"
    sc.view_settings.exposure = -0.2
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.58, 0.62, 1)
    bg.inputs["Strength"].default_value = 0.15


def add_lights(z=3.2, energy=65, panels=False):
    """Luminárias do teto (com panels, a placa acesa aparece na imagem)."""
    for x in np.arange(1.0, 14.0, 2.2):
        for y in np.arange(-1.0, 13.0, 2.6):
            light = bpy.data.lights.new("Luminária", "AREA")
            light.shape, light.size, light.size_y = "RECTANGLE", 1.2, 0.3
            light.energy = energy
            light.color = (1.0, 0.97, 0.92)
            lo = bpy.data.objects.new("Luminária", light)
            lo.location = (x, y, z)
            bpy.context.scene.collection.objects.link(lo)
            if panels:
                p = Builder()
                p.box("luz", -0.6, -0.15, 0.01, 0.6, 0.15, 0.03)
                p.build("Placa da luminária", Matrix.Translation((x, y, z)))


def main():
    ap = argparse.ArgumentParser(prog="store.py")
    ap.add_argument("run_dir", type=Path, help="pasta da rodada, com o trajectories.json")
    ap.add_argument("--frame", type=int, default=279, help="quadro do vídeo com as pessoas (279 = 9,3 s)")
    ap.add_argument("--lang", choices=["pt", "en"], default="pt", help="idioma das placas")
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--preview", action="store_true", help="Workbench em vez de Cycles: segundos em vez de minutos")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    build(args.lang)
    sc = bpy.context.scene
    coll = sc.collection
    # pessoas e olhares do quadro (o olhar termina onde o raio da câmera bate na loja)
    people = json.loads((args.run_dir / "trajectories.json").read_text())["people"]
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    k = 0
    for pid, recs in people.items():
        p = next((r for r in recs if r["frame"] == args.frame), None)
        if p is None:
            continue
        x, y, z0 = LADDER.get(pid, (*p["feet"], 0.0))
        feet = Vector((x, y, z0))
        head = feet + Vector((0, 0, 1.60))
        target = None
        if "gaze_ray" in p:
            hit, loc, *_ = sc.ray_cast(depsgraph, eye_cam, Vector(p["gaze_ray"]), distance=40)
            target = loc if hit else None
        face = (target - head).to_2d() if target else -feet.to_2d()  # sem olhar: virado para a câmera
        rot = Matrix.Rotation(math.atan2(face.y, face.x), 4, "Z")
        person(pid, k).build(f"Pessoa {pid}", Matrix.Translation(feet) @ rot)
        k += 1
        if pid in LADDER:
            ladder(z0).build("Escada", Matrix.Translation((x, y, 0)) @ rot)
        if target:
            d = target - head
            bb = Builder()
            bb.cyl("olhar", 0, 0, 0, d.length, 0.012, n=8)
            bb.sphere("olhar", 0, 0, d.length, 0.06)
            bb.build(f"Olhar {pid}", Matrix.Translation(head) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4())

    # câmeras: CCTV calibrada (sem distorção, com sobra; warp.py aplica a lente depois)
    Mc = Matrix(CAM["M"])
    cctv_cam = bpy.data.cameras.new("CCTV")
    cctv_cam.sensor_fit = "HORIZONTAL"
    cctv_cam.angle = 2 * math.atan(calib.OVERSCAN)
    cctv_cam.clip_end = 100
    cctv = bpy.data.objects.new("CCTV", cctv_cam)
    coll.objects.link(cctv)
    right, down, fwd = Mc.col[0], Mc.col[1], Mc.col[2]
    cctv.matrix_world = Matrix((
        (right.x, -down.x, -fwd.x, 0), (right.y, -down.y, -fwd.y, 0), (right.z, -down.z, -fwd.z, CAM["h"]), (0, 0, 0, 1),
    ))
    orbit = look("Órbita", (-2.5, -6.5, 8.0), (7.5, 5.0, 0.3), lens_deg=58)
    aisle = look("Corredor", (1.2, 2.4, 1.65), (8.0, 4.2, 1.0), lens_deg=70)
    plan = look("Planta", (6.8, 5.8, 25), (6.8, 5.8001, 0), ortho=31)
    plan.rotation_euler = (0, 0, 0)

    out = args.run_dir / f"loja_{args.lang}"
    out.mkdir(parents=True, exist_ok=True)
    w_raw = 1650
    h_raw = round(w_raw * 0.40 / calib.OVERSCAN)

    def shot(camera, name, res=(1280, 720)):
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.camera = camera
        sc.render.filepath = str(out / name)
        bpy.ops.render.render(write_still=True)

    if args.preview:
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.display.shading.light = "STUDIO"
        sc.display.shading.color_type = "MATERIAL"
        sc.display.shading.show_cavity = True
    else:
        use_cycles(args.samples)
        add_lights()
    shot(cctv, "cctv_raw.png", (w_raw, h_raw))
    shot(orbit, "orbita.png")
    shot(aisle, "corredor.png")
    shot(plan, "planta.png")
    sc.camera = orbit
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "loja.blend"))


if __name__ == "__main__":
    main()
