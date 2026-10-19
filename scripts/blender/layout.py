"""Planta da loja: zonas, valores iniciais dos móveis e geometria das faces. Sem bpy.

Coordenadas da loja: x ao longo das gôndolas, y atravessando a loja, z para cima;
a câmera de segurança fica sobre a origem. Cada móvel é a linha da face visível:
início (x, y), direção (graus) e comprimento; o corpo cresce à esquerda da face
(para longe da câmera), com profundidade e altura.

calib.py parte de INITIAL e grava o ajuste em config/layout_acti.json; FIXTURES
usa o ajuste quando ele existe.
"""

import json
import math
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "config"
CAMERA_FILE = CONFIG / "camera_acti.json"
LAYOUT_FILE = CONFIG / "layout_acti.json"

# zona, tipo, início da face (x, y), direção (graus), comprimento, profundidade, altura.
# 0 = gôndola, -90 = parede de fundo. A parede dos óculos não é paralela às gôndolas:
# os 60° vêm do ponto de fuga dela na foto. Nomes com "_" são móveis fora das zonas.
INITIAL = [
    ("MERCEARIA", "gondola", (3.05, 1.05), 0, 3.6, 0.9, 1.5),
    ("PONTA DE GÔNDOLA", "stand", (2.4, 1.6), -90, 0.6, 0.4, 1.3),
    ("DOCES", "gondola", (2.5, 2.75), 0, 4.0, 0.9, 1.5),
    ("SNACKS", "gondola", (4.0, 4.6), 0, 3.6, 0.9, 1.6),
    ("GELADEIRAS", "fridge", (9.3, 6.4), -90, 3.8, 0.8, 2.1),
    ("CERVEJAS", "fridge", (8.0, 2.3), -90, 3.5, 0.8, 2.1),
    ("VINHOS", "wine", (7.4, 3.3), -90, 0.9, 0.4, 1.8),
    ("ÓCULOS", "wallrack", (2.4, 4.7), 60.044, 4.2, 0.4, 1.8),
    ("CAFÉ", "counter", (7.0, 9.0), -90, 3.0, 0.7, 0.95),
    ("_EXPOSITOR", "display", (2.0, 3.65), -90, 0.9, 0.45, 1.1),  # madeira, na frente dos doces
    ("_BEBIDAS", "shelf", (10.0, 4.0), -90, 1.5, 0.5, 1.8),  # estante no fundo, ao centro
    ("_AGUA", "water", (9.0, 9.0), -90, 0.8, 0.6, 1.0),  # fardos de água perto do café
    ("_SORVETE", "freezer", (9.0, 11.0), -90, 0.8, 0.7, 0.9),
]

# Atenção por zona, em % do total (relatório do vídeo de demonstração,
# docs/experiments/2026-10-14-promo-v2.md): dá a cor e o número da placa de cada zona
ATTENTION = {
    "DOCES": 31.5, "ÓCULOS": 30.9, "SNACKS": 14.1, "CAFÉ": 5.5, "PONTA DE GÔNDOLA": 4.9,
    "VINHOS": 3.9, "GELADEIRAS": 3.7, "CERVEJAS": 3.1, "MERCEARIA": 2.3,
}
# os mesmos nomes do render_promo.py --lang en
NAMES_EN = {
    "DOCES": "Candy and chocolate", "PONTA DE GÔNDOLA": "Endcap", "MERCEARIA": "Grocery",
    "GELADEIRAS": "Beverage coolers", "VINHOS": "Wine and spirits", "CERVEJAS": "Beer",
    "SNACKS": "Back-wall snacks", "CAFÉ": "Coffee and juice", "ÓCULOS": "Sunglasses and accessories",
}


def zone_names(lang="pt"):
    """Nome de cada zona ("short" do zones_acti.json) no idioma pedido."""
    zones = json.loads((CONFIG / "zones_acti.json").read_text())["zones"]
    return {z["short"]: NAMES_EN[z["short"]] if lang == "en" else z["name"] for z in zones}


def _apply_fit():
    fit = json.loads(LAYOUT_FILE.read_text()) if LAYOUT_FILE.exists() else {}
    out = []
    for zone, kind, start, angle, L, d, H in INITIAL:
        if zone in fit:
            p = fit[zone]
            start, angle, L, d, H = (p["x"], p["y"]), p["a"], p["L"], p["d"], p["H"]
        out.append((zone, kind, start, angle, L, d, H))
    return out


FIXTURES = _apply_fit()


def face_frame(start, angle, length):
    """Origem, eixo ao longo da face e eixo do corpo (à esquerda dele), no mundo."""
    a = math.radians(angle)
    along = (math.cos(a), math.sin(a))
    return tuple(start), along, (-along[1], along[0])


def box_corners(start, angle, L, d, H):
    p0, a, b = face_frame(start, angle, L)
    return [(p0[0] + a[0] * x + b[0] * y, p0[1] + a[1] * x + b[1] * y, z)
            for z in (0, H) for x, y in ((0, 0), (L, 0), (L, d), (0, d))]
