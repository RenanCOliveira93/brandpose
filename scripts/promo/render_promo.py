"""Renderiza o vídeo promocional a partir dos dados extraídos com a configuração V2.

Linha do tempo (30 fps, 1920x1080), com `freeze` = instante do congelamento:
  raw       câmera crua com títulos de abertura
  scan      linha de varredura transforma a cena em "modo IA"
  live      análise ao vivo 1x: feixes do olhar, calor nas prateleiras, ranking
  freeze3d  tempo congela e a loja vira uma nuvem de pontos 3D (bullet time)
  ff        o resto do vídeo acelerado; o calor continua acumulando
  report    mapa de calor final + ranking de zonas
  end       encerramento

Todos os números mostrados vêm dos dados extraídos (nada é inventado).

Uso:
  python render_promo.py DATA_DIR OUT.mp4 [--freeze 19.0]
  python render_promo.py DATA_DIR OUT_DIR --stills 3,9,15,24,33,40,46
"""

import argparse
import json
import math
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

W, H, FPS = 1920, 1080, 30
FONTS = "/System/Library/Fonts"
EDGE = 32  # olhar a menos de 32 px da borda é tratado como "fora da imagem"

# Textos do HUD e do encerramento (main() troca para a versão "projeto pessoal" com --author)
CREDITS = {"title": "BRANDPOSE", "sub": "ATTENTION ANALYTICS  ·  CAM 01", "author": None, "link": None,
           "tagline": "Um projeto pessoal sobre a atenção do cliente no varejo", "watermark": False}

# Textos na tela, por idioma (main() escolhe com --lang). As chaves com {} são preenchidas com .format().
TEXTS = {
    "pt": {
        "dec": ",", "thousands": ".",
        "staff": "EQUIPE", "mode_live": "ANÁLISE QUADRO A QUADRO", "mode_3d": "MODO 3D", "mode_ff": "AVANÇO RÁPIDO",
        "scene": "CENA", "panel": "ATENÇÃO POR ZONA", "panel_foot": "{n} olhares estimados  ·  {now} pessoas agora",
        "raw1": "Toda loja tem câmeras.", "raw2": "Mas elas só gravam.", "scan": "MAPEANDO CENA  ·  POSE  ·  OLHAR",
        "scan_title": "Até agora.", "rewind": "VOLTANDO AO INÍCIO, AGORA COM IA",
        "live1": "A IA estima para onde cada pessoa olha",
        "live1_sub": "10 vezes por segundo  ·  com uma câmera de segurança comum",
        "live2": "As prateleiras mais olhadas acendem",
        "live2_sub": "o calor soma o tempo de atenção de cada cliente em cada zona",
        "frozen": "TEMPO CONGELADO", "3d": "A loja em 3D",
        "3d_sub": "profundidade estimada por IA a partir de uma única imagem",
        "ff": "{s} segundos de loja", "ff_sub": "o calor de atenção continua acumulando",
        "report": "O que mais chamou a atenção",
        "report_sub": "{s} s de vídeo  ·  uma câmera  ·  nenhum sensor extra",
        "ranking": "RANKING DE ATENÇÃO", "rank_row": "{s} s de atenção  ·  {n} pessoas",
        "kpi_gazes": "olhares estimados", "kpi_secs": "segundos de atenção", "kpi_zones": "zonas com atenção",
        "notes": ["Equipe de reposição excluída da contagem", "Olhares para fora da imagem não contam",
                  "Fragmentos da mesma pessoa unificados"],
        "end_tagline": "Atenção do cliente, em dados.",
        "end_credits": "Protótipo  ·  Gaze-LLE + YOLO11 + Depth Anything V2  ·  vídeo de demonstração ACTi",
        "created_by": "CRIADO POR",
        "personal_title": "ATENÇÃO DO CLIENTE", "personal_sub": "PROJETO PESSOAL  ·  CAM 01",
        "zones": {},  # os nomes do zones.json já estão em português
    },
    "en": {
        "dec": ".", "thousands": ",",
        "staff": "STAFF", "mode_live": "FRAME-BY-FRAME ANALYSIS", "mode_3d": "3D MODE", "mode_ff": "FAST FORWARD",
        "scene": "SCENE", "panel": "ATTENTION BY ZONE", "panel_foot": "{n} estimated gazes  ·  {now} people now",
        "raw1": "Every store has cameras.", "raw2": "But they only record.", "scan": "MAPPING SCENE  ·  POSE  ·  GAZE",
        "scan_title": "Until now.", "rewind": "BACK TO THE START, NOW WITH AI",
        "live1": "AI estimates where each person looks",
        "live1_sub": "10 times per second  ·  with an ordinary security camera",
        "live2": "The most-viewed shelves light up",
        "live2_sub": "the heat adds up each shopper's attention time in each zone",
        "frozen": "TIME FROZEN", "3d": "The store in 3D",
        "3d_sub": "depth estimated by AI from a single image",
        "ff": "{s} seconds in the store", "ff_sub": "the attention heat keeps building up",
        "report": "What drew the most attention",
        "report_sub": "{s} s of video  ·  one camera  ·  no extra sensors",
        "ranking": "ATTENTION RANKING", "rank_row": "{s} s of attention  ·  {n} people",
        "kpi_gazes": "estimated gazes", "kpi_secs": "seconds of attention", "kpi_zones": "zones with attention",
        "notes": ["Restocking staff excluded from the count", "Gazes leaving the frame do not count",
                  "Fragments of the same person merged"],
        "end_tagline": "Shopper attention, in data.",
        "end_credits": "Prototype  ·  Gaze-LLE + YOLO11 + Depth Anything V2  ·  ACTi demo video",
        "created_by": "CREATED BY",
        "personal_title": "SHOPPER ATTENTION", "personal_sub": "PERSONAL PROJECT  ·  CAM 01",
        "tagline": "A personal project on shopper attention in retail",
        # "short" do zones.json -> (rótulo curto, nome completo)
        "zones": {"DOCES": ("CANDY", "Candy and chocolate"), "PONTA DE GÔNDOLA": ("ENDCAP", "Endcap"),
                  "MERCEARIA": ("GROCERY", "Grocery"), "GELADEIRAS": ("COOLERS", "Beverage coolers"),
                  "VINHOS": ("WINE", "Wine and spirits"), "CERVEJAS": ("BEER", "Beer"),
                  "SNACKS": ("SNACKS", "Back-wall snacks"), "CAFÉ": ("COFFEE", "Coffee and juice"),
                  "ÓCULOS": ("SUNGLASSES", "Sunglasses and accessories")},
    },
}
TX = TEXTS["pt"]


def zone_text(zone, key="short"):
    """Nome da zona no idioma do vídeo (a lógica continua usando o "short" original)."""
    tr = TX["zones"].get(zone["short"])
    return zone[key] if tr is None else tr[0 if key == "short" else 1]

# Cores (BGR para OpenCV)
CYAN = np.array([255, 225, 70], np.float32)
CYAN_DIM = np.array([170, 140, 40], np.float32)
AMBER = (40, 165, 255)
AMBER_CORE = (175, 235, 255)
SKEL = (255, 240, 165)
# Cores (RGB para PIL)
RGB_WHITE = (255, 255, 255)
RGB_CYAN = (80, 225, 255)
RGB_AMBER = (255, 170, 50)
RGB_DIM = (150, 170, 185)
RGB_RED = (255, 70, 70)

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16)]


# ---------------------------------------------------------------- utilidades

@lru_cache(maxsize=None)
def font(kind, size):
    path, index = {
        "bold": ("Avenir Next.ttc", 0), "heavy": ("Avenir Next.ttc", 8), "demi": ("Avenir Next.ttc", 2),
        "medium": ("Avenir Next.ttc", 5), "regular": ("Avenir Next.ttc", 7),
        "mono": ("Menlo.ttc", 0), "monob": ("Menlo.ttc", 1),
        "num": ("Supplemental/DIN Condensed Bold.ttf", 0),
    }[kind]
    return ImageFont.truetype(f"{FONTS}/{path}", size, index=index)


def clamp01(x):
    return max(0.0, min(1.0, x))


def ease_out(u):
    u = clamp01(u)
    return 1 - (1 - u) ** 3


def ease_in_out(u):
    u = clamp01(u)
    return u * u * (3 - 2 * u)


def fade(t, t0, t1, fin=0.35, fout=0.35):
    """Opacidade de um elemento visível em [t0, t1] com entrada/saída suaves."""
    if t < t0 or t > t1:
        return 0.0
    return min(ease_out((t - t0) / fin) if fin > 0 else 1.0, ease_out((t1 - t) / fout) if fout > 0 else 1.0)


def fmt_int(n):
    return f"{int(round(n)):,}".replace(",", TX["thousands"])


def fmt_dec(x, nd=1):
    return f"{x:.{nd}f}".replace(".", TX["dec"])


def glow(img_u8, sigma=5.0, scale=4):
    small = cv2.resize(img_u8, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class TextLayer:
    """Camada RGBA para texto e elementos de interface (composição alfa correta)."""

    def __init__(self):
        self.im = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    def text(self, xy, s, fnt, rgb=RGB_WHITE, a=1.0, anchor="la", track=0, shadow=False):
        if a <= 0.01 or not s:
            return 0
        d0 = ImageDraw.Draw(self.im)
        if track:
            widths = [d0.textlength(ch, font=fnt) + track for ch in s]
            total = sum(widths) - track
        else:
            total = d0.textlength(s, font=fnt)
        asc, desc = fnt.getmetrics()
        pad = 6
        tile = Image.new("RGBA", (int(total) + 2 * pad + 4, asc + desc + 2 * pad), (0, 0, 0, 0))
        dt = ImageDraw.Draw(tile)
        if track:
            x = pad
            for ch, wch in zip(s, widths):
                dt.text((x, pad), ch, font=fnt, fill=(*rgb, 255))
                x += wch
        else:
            dt.text((pad, pad), s, font=fnt, fill=(*rgb, 255))
        if shadow:
            sh = Image.new("RGBA", tile.size, (0, 0, 0, 0))
            sh.putalpha(tile.getchannel("A").point(lambda v: int(v * 0.65)))
            tile = Image.alpha_composite(sh.transform(sh.size, Image.AFFINE, (1, 0, -2, 0, 1, -2)), tile)
        if a < 1:
            tile.putalpha(tile.getchannel("A").point(lambda v: int(v * a)))
        x, y = xy
        if anchor[0] == "r":
            x -= total
        elif anchor[0] == "m":
            x -= total / 2
        if anchor[1] == "m":
            y -= (asc + desc) / 2
        elif anchor[1] == "s":
            y -= asc
        elif anchor[1] == "b":
            y -= asc + desc
        self.im.alpha_composite(tile, dest=(int(x - pad), int(y - pad)))
        return total

    def box(self, xyxy, fill=None, outline=None, width=1, radius=6):
        x0, y0, x1, y1 = (int(v) for v in xyxy)
        if x1 <= x0 or y1 <= y0:
            return
        tile = Image.new("RGBA", (x1 - x0 + 1, y1 - y0 + 1), (0, 0, 0, 0))
        ImageDraw.Draw(tile).rounded_rectangle((0, 0, x1 - x0, y1 - y0), radius, fill=fill, outline=outline, width=width)
        self.im.alpha_composite(tile, dest=(x0, y0))

    def composite(self, img):
        bb = self.im.getbbox()
        if not bb:
            return img
        x0, y0, x1, y1 = bb
        arr = np.asarray(self.im.crop(bb)).astype(np.float32)
        a = arr[..., 3:4] / 255.0
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (roi * (1 - a) + arr[..., 2::-1] * a).astype(np.uint8)
        return img


def glass(img, xyxy, dark=0.62):
    """Painel de vidro fosco com cantos em colchete."""
    x0, y0, x1, y1 = xyxy
    roi = img[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(roi, (0, 0), 14).astype(np.float32)
    tint = np.array([28, 20, 10], np.float32)
    img[y0:y1, x0:x1] = (blurred * (1 - dark) + tint * dark).astype(np.uint8)
    c = tuple(int(v) for v in CYAN_DIM)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 75, 30), 1)
    L = 16
    for (px, py, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * L, py), c, 2)
        cv2.line(img, (px, py), (px, py + dy * L), c, 2)


# ---------------------------------------------------------------- dados

class Data:
    def __init__(self, D):
        self.D = Path(D)
        self.meta = json.loads((self.D / "meta.json").read_text())
        self.tracks = json.loads((self.D / "tracks.json").read_text())
        self.gaze = json.loads((self.D / "gaze.json").read_text())
        self.hms = np.load(self.D / "gaze_heatmaps.npy")
        self.masks = np.load(self.D / "person_masks_packed.npy")
        self.zcfg = json.loads((self.D / "zones.json").read_text())
        self.zones = self.zcfg["zones"]
        self.n_frames = self.meta["frames"]
        self.every = self.meta["gaze_every"]
        self.ds = self.meta["hm_ds"]
        self.mask_shape = tuple(self.meta["mask_shape"])
        self.by_frame = [{p["id"]: p for p in fr} for fr in self.tracks]
        self._zone_maps()
        self._staff()
        self._merge_tracks()
        self._gaze_tracks()
        self._stats()

    # Junta fragmentos da mesma pessoa (ID novo que começa logo depois e perto de
    # onde outro terminou) e numera os clientes em ordem de chegada.
    def _merge_tracks(self):
        span = {}
        for f, fr in enumerate(self.tracks):
            for p in fr:
                b = p["box"]
                pos = ((b[0] + b[2]) / 2, b[3])
                if p["id"] not in span:
                    span[p["id"]] = [f, f, pos, pos]
                span[p["id"]][1] = f
                span[p["id"]][3] = pos
        order = sorted((s[0], tid) for tid, s in span.items())
        parent = {}
        taken = set()
        for f0, tid in order:
            parent[tid] = tid
            if tid in self.staff:
                continue
            best = None
            for f0b, pid in order:
                if pid == tid or pid in self.staff or pid in taken:
                    continue
                e = span[pid][1]
                gap = f0 - e
                dist = math.dist(span[pid][3], span[tid][2])
                if 0 <= gap <= 45 and dist < 150 and (best is None or gap < best[0]):
                    best = (gap, pid)
            if best:
                taken.add(best[1])
                parent[tid] = parent[best[1]]
        self.person = parent  # track -> id da pessoa (primeiro track da cadeia)
        firsts = sorted({(span[parent[t]][0], parent[t]) for t in span if t not in self.staff})
        self.number = {pid: i + 1 for i, (_, pid) in enumerate(firsts)}
        self.n_customers = len(firsts)

    # Zonas: mapa de rótulos em 1/4 da resolução, com tolerância de 40 px
    def _zone_maps(self):
        order = ["GELADEIRAS", "CERVEJAS", "VINHOS", "SNACKS", "CAFÉ", "ÓCULOS", "DOCES", "MERCEARIA", "PONTA DE GÔNDOLA"]
        idx = {z["short"]: i for i, z in enumerate(self.zones)}
        zmap = -np.ones((H // 4, W // 4), np.int16)
        self.zone_full = []
        for name in order:
            i = idx[name]
            tmp = np.zeros((H // 4, W // 4), np.uint8)
            for p in self.zones[i]["polys"]:
                cv2.fillPoly(tmp, [np.array(p, np.int32) // 4], 1)
            zmap[tmp == 1] = i
        dist, (iy, ix) = ndimage.distance_transform_edt(zmap < 0, return_indices=True)
        znear = zmap[iy, ix]
        znear[dist > 10] = -1
        self.zmap = znear
        half = np.zeros((H // 2, W // 2), np.uint8)
        for z in self.zones:
            for p in z["polys"]:
                cv2.fillPoly(half, [np.array(p, np.int32) // 2], 255)
        self.zone_soft = cv2.GaussianBlur(half, (0, 0), 4).astype(np.float32) / 255.0
        # Máscara suave de cada zona (1/4 da resolução) para o brilho da zona ativa
        self.zone_q = []
        for z in self.zones:
            q = np.zeros((H // 4, W // 4), np.uint8)
            for p in z["polys"]:
                cv2.fillPoly(q, [np.array(p, np.int32) // 4], 255)
            self.zone_q.append(cv2.GaussianBlur(q, (0, 0), 3).astype(np.float32) / 255.0)
        self.centroids = []
        for z in self.zones:
            p = np.array(z["polys"][0], np.float32)
            self.centroids.append(p.mean(0))

    def zone_at(self, x, y):
        xi, yi = int(np.clip(x / 4, 0, W // 4 - 1)), int(np.clip(y / 4, 0, H // 4 - 1))
        return int(self.zmap[yi, xi])

    def on_person(self, f, x, y, r=4):
        """O alvo do olhar cai sobre alguém (silhueta do quadro f, meia resolução)?"""
        m = self._mask_cached(f)
        xi, yi = int(np.clip(x / 2, 0, m.shape[1] - 1)), int(np.clip(y / 2, 0, m.shape[0] - 1))
        return bool(m[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1].any())

    def zone_for(self, f, x, y):
        """Zona olhada no quadro f; -1 se o alvo estiver sobre uma pessoa ou fora das zonas."""
        z = self.zone_at(x, y)
        if z >= 0 and self.on_person(f, x, y):
            return -1
        return z

    def _mask_cached(self, f):
        if getattr(self, "_mk", None) != f:
            self._mk, self._mv = f, self.mask(f)
        return self._mv

    def _staff(self):
        sx0, sy0 = self.zcfg["staff_heads"][0]
        sx1, sy1 = self.zcfg["staff_heads"][2]
        heads = defaultdict(list)
        for fr in self.tracks:
            for p in fr:
                if p["head"]:
                    x1, y1, x2, y2 = p["head"]
                    heads[p["id"]].append(((x1 + x2) / 2, (y1 + y2) / 2))
        self.staff = set()
        for tid, hs in heads.items():
            hx, hy = np.median(np.array(hs), axis=0)
            if sx0 <= hx <= sx1 and sy0 <= hy <= sy1:
                self.staff.add(tid)

    # Olhar: suaviza mapas de calor vizinhos, suprime a própria cabeça, refina o pico
    def _gaze_tracks(self):
        by = defaultdict(list)
        for g in self.gaze:
            by[g["id"]].append(g)
        self.gt = {}
        hs, ws = self.hms.shape[1:]
        for tid, gs in by.items():
            gs.sort(key=lambda g: g["frame"])
            fr = np.array([g["frame"] for g in gs])
            xs, ys, ios = [], [], []
            for i, g in enumerate(gs):
                acc = self.hms[g["hm"]].astype(np.float32) * 0.5
                io = g["inout"] * 0.5
                wsum = 0.5
                for j, w in ((i - 1, 0.25), (i + 1, 0.25)):
                    if 0 <= j < len(gs) and abs(fr[j] - fr[i]) <= self.every:
                        acc += self.hms[gs[j]["hm"]].astype(np.float32) * w
                        io += gs[j]["inout"] * w
                        wsum += w
                acc /= wsum
                io /= wsum
                head = self.by_frame[fr[i]].get(tid, {}).get("head")
                if head:
                    x1, y1, x2, y2 = head
                    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.75, (y2 - y1) * 0.75
                    acc[int(max(0, cy - hh) / self.ds):int((cy + hh) / self.ds) + 1,
                        int(max(0, cx - hw) / self.ds):int((cx + hw) / self.ds) + 1] = 0
                py, px = np.unravel_index(acc.argmax(), acc.shape)
                y0, y1_, x0, x1_ = max(0, py - 1), min(hs, py + 2), max(0, px - 1), min(ws, px + 2)
                win = acc[y0:y1_, x0:x1_]
                gy, gx = np.mgrid[y0:y1_, x0:x1_]
                s = win.sum() + 1e-9
                xs.append(((gx * win).sum() / s + 0.5) * self.ds)
                ys.append(((gy * win).sum() / s + 0.5) * self.ds)
                ios.append(io)
            xs, ys, ios = np.array(xs), np.array(ys), np.array(ios)
            # Pico colado na borda = alvo fora da imagem (o recorte é limitado ao quadro).
            # Esses olhares não contam como atenção em nenhuma zona.
            edge = (xs < EDGE) | (xs > W - EDGE) | (ys < EDGE) | (ys > H - EDGE)
            ios = np.where(edge, 0.0, ios)
            # Suavização das posições em trechos contínuos (pesos 1-2-3-2-1)
            kx, ky = xs.copy(), ys.copy()
            wts = {-2: 1, -1: 2, 0: 3, 1: 2, 2: 1}
            for i in range(len(gs)):
                sx = sy = sw = 0.0
                for k, w in wts.items():
                    j = i + k
                    if 0 <= j < len(gs) and abs(fr[j] - fr[i]) <= abs(k) * self.every:
                        sx += xs[j] * w
                        sy += ys[j] * w
                        sw += w
                kx[i], ky[i] = sx / sw, sy / sw
            self.gt[tid] = {"f": fr, "x": kx, "y": ky, "io": ios}

    def gaze_at(self, tid, f):
        g = self.gt.get(tid)
        if g is None:
            return None
        fr = g["f"]
        k = np.searchsorted(fr, f, side="right") - 1
        if k >= 0 and fr[k] == f:
            return g["x"][k], g["y"][k], g["io"][k]
        if k >= 0 and k + 1 < len(fr) and fr[k + 1] - fr[k] <= 2 * self.every:
            a = (f - fr[k]) / (fr[k + 1] - fr[k])
            return tuple(g[c][k] * (1 - a) + g[c][k + 1] * a for c in ("x", "y", "io"))
        if k >= 0 and f - fr[k] <= self.every:
            return g["x"][k], g["y"][k], g["io"][k]
        return None

    # Estatísticas acumuladas por quadro (só clientes, olhar dentro da imagem)
    def _stats(self):
        nz = len(self.zones)
        dt = self.every / self.meta["fps"]
        inc = np.zeros((self.n_frames, nz), np.float32)
        cnt = np.zeros(self.n_frames, np.int32)
        self.samples = []  # (frame, x, y, zona)
        self.dwell = {}  # (tid, frame) -> (zona, segundos seguidos na zona)
        viewers = defaultdict(set)
        for tid, g in self.gt.items():
            if tid in self.staff:
                continue
            run_z, run_t = -1, 0.0
            for f, x, y, io in zip(g["f"], g["x"], g["y"], g["io"]):
                if io < 0.5:
                    run_z, run_t = -1, 0.0
                    continue
                z = self.zone_for(int(f), x, y)
                cnt[f] += 1
                self.samples.append((int(f), float(x), float(y), z))
                if z >= 0:
                    inc[f, z] += dt
                    viewers[z].add(self.person[tid])
                    run_t = run_t + dt if z == run_z else dt
                    run_z = z
                else:
                    run_z, run_t = -1, 0.0
                self.dwell[(tid, int(f))] = (run_z, run_t)
        self.cum_zone = np.cumsum(inc, axis=0)
        self.cum_count = np.cumsum(cnt)
        self.viewers = {z: len(v) for z, v in viewers.items()}
        self.samples.sort()
        self.sample_frames = np.array([s[0] for s in self.samples])

    def dwell_at(self, tid, f):
        g = self.gt.get(tid)
        if g is None:
            return None
        fr = g["f"]
        k = np.searchsorted(fr, f, side="right") - 1
        if k < 0:
            return None
        return self.dwell.get((tid, int(fr[k])))

    def mask(self, f):
        bits = np.unpackbits(self.masks[min(f, len(self.masks) - 1)])
        return bits[: self.mask_shape[0] * self.mask_shape[1]].reshape(self.mask_shape)


class Heat:
    """Mapa de calor acumulado (meia resolução), restrito às zonas."""

    def __init__(self, data, sigma=12):
        self.d = data
        r = int(3 * sigma)
        g = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        self.k = np.outer(g, g)
        self.r = r
        self.buf = np.zeros((H // 2, W // 2), np.float32)
        self.pos = 0  # quantas amostras já foram somadas
        full = self._accumulate_all()
        vals = full[self.d.zone_soft > 0.5]
        self.scale = float(np.percentile(vals[vals > 0], 99.5)) / 2.6 if (vals > 0).any() else 1.0
        self.buf[:] = 0
        self.pos = 0
        lut = np.zeros((256, 3), np.float32)
        anchors = [(0.0, (0, 0, 0)), (0.18, (70, 12, 95)), (0.42, (175, 25, 120)), (0.62, (245, 70, 60)),
                   (0.8, (255, 155, 35)), (1.0, (255, 245, 195))]
        for i in range(256):
            t = i / 255
            for (t0, c0), (t1, c1) in zip(anchors, anchors[1:]):
                if t0 <= t <= t1:
                    a = (t - t0) / (t1 - t0)
                    rgb = [c0[j] * (1 - a) + c1[j] * a for j in range(3)]
                    lut[i] = rgb[::-1]
                    break
        self.lut = lut

    def _add(self, x, y, w=1.0):
        cx, cy = int(x / 2), int(y / 2)
        r = self.r
        x0, x1, y0, y1 = cx - r, cx + r + 1, cy - r, cy + r + 1
        hh, ww = self.buf.shape
        kx0, ky0 = max(0, -x0), max(0, -y0)
        kx1, ky1 = self.k.shape[1] - max(0, x1 - ww), self.k.shape[0] - max(0, y1 - hh)
        if kx1 <= kx0 or ky1 <= ky0:
            return
        self.buf[max(0, y0):min(hh, y1), max(0, x0):min(ww, x1)] += w * self.k[ky0:ky1, kx0:kx1]

    def _accumulate_all(self):
        for f, x, y, z in self.d.samples:
            if z >= 0:
                self._add(x, y)
        self.pos = len(self.d.samples)
        return self.buf.copy()

    def at(self, f):
        """Avança (ou refaz) o acúmulo até o quadro f e devolve o valor normalizado [0,1]."""
        target = int(np.searchsorted(self.d.sample_frames, f, side="right"))
        if target < self.pos:
            self.buf[:] = 0
            self.pos = 0
        for i in range(self.pos, target):
            f_, x, y, z = self.d.samples[i]
            if z >= 0:
                self._add(x, y)
        self.pos = target
        return (1 - np.exp(-self.buf / self.scale)) * self.d.zone_soft

    def colorize(self, v_half, people_full=None, gain=1.0):
        idx = np.clip(v_half * 255, 0, 255).astype(np.uint8)
        col = self.lut[idx] * (np.clip(v_half * 1.6, 0, 1)[..., None])
        col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
        if people_full is not None:
            col *= (1 - people_full[..., None])
        g = glow(np.clip(col, 0, 255).astype(np.uint8), sigma=7, scale=4)
        return col * gain + g * 0.7 * gain


# ---------------------------------------------------------------- cena

class Scene:
    def __init__(self, data, video):
        self.d = data
        self.cap = cv2.VideoCapture(video)
        self.cur = -1
        self.frame = None
        plate = cv2.imread(str(data.D / "clean_plate.png"))
        self.plate = plate
        g = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
        edges = cv2.Canny(g, 45, 130)
        e = edges.astype(np.float32)[..., None] / 255.0 * CYAN * 0.32
        self.edges = e + glow(np.clip(e * 2, 0, 255).astype(np.uint8), sigma=3, scale=2) * 0.35
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        self.vignette = (1 - 0.32 * r2).astype(np.float32)[..., None]
        scan = np.ones((H, 1), np.float32)
        scan[::4] = 0.86
        self.scanlines = scan[..., None]
        self.tint = np.array([1.08, 0.97, 0.66], np.float32)
        self.heat = Heat(data)

    def src(self, f):
        f = int(np.clip(f, 0, self.d.n_frames - 1))
        if f == self.cur:
            return self.frame
        if f < self.cur or f > self.cur + 40:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        else:
            while self.cur < f - 1:
                self.cap.grab()
                self.cur += 1
        ok, fr = self.cap.read()
        if ok:
            self.frame = fr
        self.cur = f
        return self.frame

    def people_soft(self, f):
        m = cv2.resize(self.d.mask(f).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(m, (0, 0), 1.5)

    def analysis_base(self, frame, darkness=0.36):
        fr = frame.astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        base = (0.72 * gray * self.tint + 0.28 * fr) * darkness
        return base * self.scanlines * self.vignette + self.edges

    def live(self, f, t, opts):
        """Composição do "modo IA" para o quadro f. Devolve (img_u8, TextLayer)."""
        frame = self.src(f)
        base = self.analysis_base(frame)
        m = self.people_soft(f)
        if opts.get("people", True):
            fr = frame.astype(np.float32)
            base = base * (1 - m[..., None] * 0.9) + fr * 0.95 * (m[..., None] * 0.9)
            rim = cv2.dilate(m, np.ones((7, 7), np.uint8)) - cv2.erode(m, np.ones((5, 5), np.uint8))
            rim_u8 = (np.clip(rim, 0, 1)[..., None] * CYAN).astype(np.uint8)
            base += rim_u8.astype(np.float32) * 0.55 + glow(rim_u8, 4, 4) * 0.9
        heat_v = self.heat.at(f)
        base += self.heat.colorize(heat_v, m, gain=opts.get("heat_gain", 1.0))

        ov = np.zeros((H, W, 3), np.uint8)
        tl = TextLayer()
        present = self.d.by_frame[min(f, self.d.n_frames - 1)]
        active_z = set()
        n_customers = 0
        for tid, p in present.items():
            staff = tid in self.d.staff
            if not staff and p["head"]:
                n_customers += 1
            if opts.get("skeleton", True):
                draw_skeleton(ov, p, staff)
            if p["head"] is None:
                continue
            x1, y1, x2, y2 = p["head"]
            hc = ((x1 + x2) / 2, (y1 + y2) / 2)
            if opts.get("tags", True):
                person_tag(tl, p, staff, self.d.number.get(self.d.person.get(tid, tid)))
            if staff:
                continue
            g = self.d.gaze_at(tid, f)
            if g is None:
                continue
            gx, gy, io = g
            a = clamp01((io - 0.45) / 0.15)
            if a <= 0:
                continue
            draw_beam(ov, hc, (gx, gy), a, t + tid * 0.37)
            z = self.d.zone_for(f, gx, gy)
            if z >= 0:
                active_z.add(z)
                if opts.get("labels", True):
                    dw = self.d.dwell_at(tid, f)
                    secs = dw[1] if dw and dw[0] == z else 0.0
                    target_label(tl, (gx, gy), zone_text(self.d.zones[z]), secs, a)
        if opts.get("zones", True) and active_z:
            # A zona que está sendo olhada ganha um brilho suave
            q = sum(self.d.zone_q[z] for z in active_z)
            q = cv2.resize(np.clip(q, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR) * (1 - m)
            base += q[..., None] * np.array(AMBER, np.float32) * 0.10
        out = base + ov.astype(np.float32) + glow(ov, 5, 4) * 1.1
        return np.clip(out, 0, 255).astype(np.uint8), tl, n_customers


def draw_skeleton(ov, p, staff):
    kp, kc = np.array(p["kp"]), np.array(p["kc"])
    col = (130, 130, 130) if staff else SKEL
    for a, b in COCO_EDGES:
        if kc[a] > 0.4 and kc[b] > 0.4:
            cv2.line(ov, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), col, 2, cv2.LINE_AA)
    for i in range(5, 17):
        if kc[i] > 0.4:
            cv2.circle(ov, tuple(int(v) for v in kp[i]), 3, col, -1, cv2.LINE_AA)
    if p["head"]:
        x1, y1, x2, y2 = p["head"]
        r = int(max(6, (x2 - x1) * 0.42))
        cv2.circle(ov, (int((x1 + x2) / 2), int((y1 + y2) / 2)), r, col, 1, cv2.LINE_AA)


def draw_beam(ov, o, t, a, phase):
    o = np.array(o, np.float32)
    tg = np.array(t, np.float32)
    d = tg - o
    L = float(np.linalg.norm(d))
    if L < 12:
        return
    u = d / L
    n = np.array([-u[1], u[0]], np.float32)
    w0, w1 = 3.0, float(np.clip(0.11 * L, 12, 48))
    pad = int(w1 + 30)
    x0 = int(max(0, min(o[0], tg[0]) - pad))
    y0 = int(max(0, min(o[1], tg[1]) - pad))
    x1 = int(min(W, max(o[0], tg[0]) + pad))
    y1 = int(min(H, max(o[1], tg[1]) + pad))
    if x1 <= x0 or y1 <= y0:
        return
    tmp = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    off = np.array([x0, y0], np.float32)
    N = 10
    for i in range(N):
        a0, a1 = i / N, (i + 1) / N
        p0, p1 = o + d * a0 - off, o + d * a1 - off
        wa, wb = w0 + (w1 - w0) * a0, w0 + (w1 - w0) * a1
        inten = a * (0.06 + 0.30 * a1 ** 1.3)
        pts = np.array([p0 + n * wa, p1 + n * wb, p1 - n * wb, p0 - n * wa]).astype(np.int32)
        cv2.fillConvexPoly(tmp, pts, tuple(int(c * inten) for c in AMBER), cv2.LINE_AA)
    cv2.line(tmp, tuple((o - off).astype(int)), tuple((tg - off).astype(int)),
             tuple(int(c * a * 0.85) for c in AMBER_CORE), 2, cv2.LINE_AA)
    pc = tuple((tg - off).astype(int))
    r = int(17 + 4 * math.sin(phase * 2 * math.pi * 1.3))
    cv2.circle(tmp, pc, r, tuple(int(c * a) for c in AMBER_CORE), 2, cv2.LINE_AA)
    cv2.circle(tmp, pc, 4, tuple(int(c * a) for c in AMBER_CORE), -1, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(tmp, (pc[0] + dx * (r + 4), pc[1] + dy * (r + 4)), (pc[0] + dx * (r + 12), pc[1] + dy * (r + 12)),
                 tuple(int(c * a) for c in AMBER_CORE), 2, cv2.LINE_AA)
    ov[y0:y1, x0:x1] = cv2.add(ov[y0:y1, x0:x1], tmp)


def person_tag(tl, p, staff, number):
    x1, y1, x2, y2 = p["head"]
    cx = (x1 + x2) / 2
    label = TX["staff"] if staff else f"#{number or 0:02d}"
    fnt = font("monob", 15)
    wtxt = ImageDraw.Draw(tl.im).textlength(label, font=fnt)
    bx0, by1 = cx - wtxt / 2 - 8, y1 - 10
    if by1 - 24 < 8:  # sem espaço acima da cabeça: etiqueta embaixo
        by1 = y2 + 34
    fill = (60, 60, 60, 170) if staff else (10, 60, 80, 185)
    outline = (150, 150, 150, 220) if staff else (*RGB_CYAN, 230)
    tl.box((bx0, by1 - 24, bx0 + wtxt + 16, by1), fill=fill, outline=outline, radius=5)
    tl.text((cx, by1 - 12), label, fnt, (200, 200, 200) if staff else RGB_WHITE, anchor="mm")


def target_label(tl, xy, name, secs, a):
    x, y = xy
    fnt = font("demi", 17)
    txt = f"{name}  ·  {fmt_dec(secs)} s" if secs >= 0.3 else name
    wtxt = ImageDraw.Draw(tl.im).textlength(txt, font=fnt)
    bx0, by0 = x + 26, y - 44
    if bx0 + wtxt + 20 > W - 10:
        bx0 = x - 26 - wtxt - 20
    tl.box((bx0, by0, bx0 + wtxt + 20, by0 + 28), fill=(40, 25, 5, int(190 * a)),
           outline=(*RGB_AMBER, int(235 * a)), radius=6)
    tl.text((bx0 + 10, by0 + 14), txt, fnt, (255, 225, 170), a=a, anchor="lm")


# ---------------------------------------------------------------- HUD e textos

def hud_live(tl, img, t, src_t, mode="live", badge=None):
    # Faixas escuras em cima e embaixo para legibilidade
    top = np.linspace(0.55, 0, 120, dtype=np.float32)[:, None, None]
    img[:120] = (img[:120].astype(np.float32) * (1 - top)).astype(np.uint8)
    bot = np.linspace(0, 0.6, 260, dtype=np.float32)[:, None, None]
    img[H - 260:] = (img[H - 260:].astype(np.float32) * (1 - bot)).astype(np.uint8)
    tl.text((48, 34), CREDITS["title"], font("heavy", 30), RGB_WHITE, track=7)
    tl.text((50, 74), CREDITS["sub"], font("mono", 15), RGB_CYAN, track=2)
    pulse = 0.55 + 0.45 * math.sin(t * 2 * math.pi * 0.9)
    # Não é tempo real (o Mac processa ~3 quadros/s), então não dizemos "ao vivo"
    label = TX["mode_" + mode]
    wl = tl.text((W - 48, 40), label, font("monob", 18), RGB_WHITE, anchor="ra", track=2)
    if mode == "live":
        cv2.circle(img, (int(W - 48 - wl - 22), 52), 7, (60, 60, 255), -1, cv2.LINE_AA)
        img[45:60, int(W - 48 - wl - 30):int(W - 48 - wl - 14)] = (
            img[45:60, int(W - 48 - wl - 30):int(W - 48 - wl - 14)] * (0.5 + 0.5 * pulse)).astype(np.uint8)
    tl.text((W - 48, 74), f"{TX['scene']} {int(src_t // 60):02d}:{int(src_t % 60):02d}  ·  1080p  ·  30 fps",
            font("mono", 15), RGB_DIM, anchor="ra", track=1)
    if badge:
        tl.box((W / 2 - 70, 30, W / 2 + 70, 74), fill=(255, 170, 50, 230), radius=8)
        tl.text((W / 2, 52), badge, font("monob", 24), (20, 12, 0), anchor="mm")
    # Linha do tempo do vídeo de origem
    prog = clamp01(src_t / 60.0)
    cv2.line(img, (0, H - 3), (W, H - 3), (60, 50, 25), 3)
    cv2.line(img, (0, H - 3), (int(W * prog), H - 3), tuple(int(v) for v in CYAN), 3)


def watermark(img):
    """Assinatura discreta no rodapé central (o único trecho livre em todas as cenas)."""
    tl = TextLayer()
    d0 = ImageDraw.Draw(tl.im)
    name, rest = CREDITS["author"], (f"  ·  {CREDITS['link']}" if CREDITS["link"] else "")
    fn, fr = font("demi", 22), font("medium", 22)
    total = d0.textlength(name, font=fn) + d0.textlength(rest, font=fr)
    x0, y = W / 2 - total / 2, H - 48
    w1 = tl.text((x0, y), name, fn, (240, 244, 248), a=0.68, shadow=True)
    if rest:
        tl.text((x0 + w1, y), rest, fr, (215, 228, 238), a=0.58, shadow=True)
    return tl.composite(img)


def caption(tl, t, t0, t1, main, sub=None, x=70, y=900):
    a = fade(t, t0, t1, 0.45, 0.35)
    if a <= 0:
        return
    dy = (1 - ease_out((t - t0) / 0.6)) * 18
    cv2_bar_alpha = a
    tl.box((x - 22, y - 6 + dy, x - 16, y + 52 + (44 if sub else 0) + dy), fill=(*RGB_AMBER, int(255 * cv2_bar_alpha)), radius=2)
    tl.text((x, y + dy), main, font("bold", 46), RGB_WHITE, a=a, shadow=True)
    if sub:
        tl.text((x + 2, y + 62 + dy), sub, font("medium", 26), (205, 225, 235), a=a * 0.95, shadow=True)


def zone_panel(tl, img, d, f, n_now, alpha=1.0, x0=1470, y0=690, x1=1880, y1=1040):
    if alpha <= 0:
        return
    glass(img, (x0, y0, x1, y1))
    tl.text((x0 + 22, y0 + 20), TX["panel"], font("monob", 16), RGB_CYAN, a=alpha, track=2)
    vals = d.cum_zone[min(f, d.n_frames - 1)]
    total = vals.sum()
    order = np.argsort(-vals)[:5]
    for r, zi in enumerate(order):
        yy = y0 + 60 + r * 52
        share = vals[zi] / total if total > 0 else 0.0
        tl.text((x0 + 22, yy), zone_text(d.zones[zi]), font("demi", 19), RGB_WHITE, a=alpha)
        tl.text((x1 - 22, yy - 4), f"{int(round(share * 100))}%", font("num", 30), RGB_AMBER, a=alpha, anchor="ra")
        bw = x1 - x0 - 44
        cv2.rectangle(img, (x0 + 22, yy + 30), (x0 + 22 + bw, yy + 35), (70, 55, 25), -1)
        if share > 0:
            cv2.rectangle(img, (x0 + 22, yy + 30), (x0 + 22 + int(bw * share), yy + 35), AMBER, -1)
    cnt = d.cum_count[min(f, d.n_frames - 1)]
    tl.text((x0 + 22, y1 - 34), TX["panel_foot"].format(n=fmt_int(cnt), now=n_now),
            font("mono", 14), RGB_DIM, a=alpha)


# ---------------------------------------------------------------- 3D

class Cloud:
    def __init__(self, data, scene, freeze_f, stride=2):
        D = data.D
        info = json.loads((D / "depth_info.json").read_text())
        depth = np.load(D / f"depth_freeze_{freeze_f}.npy")
        self.f = (W / 2) / math.tan(math.radians(info["hfov_deg"]) / 2)
        self.stride = stride
        frame = scene.src(freeze_f).copy()
        self.frame = frame
        ys, xs = np.mgrid[0:H:stride, 0:W:stride]
        z = depth[ys, xs]
        # Remove pontos "voadores" nas bordas de profundidade
        disc = np.zeros_like(z)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            disc = np.maximum(disc, np.abs(z - np.roll(np.roll(z, dy, 0), dx, 1)) / z)
        keep = (disc < 0.045) & (z < 20)
        xs, ys, z = xs[keep], ys[keep], z[keep]
        P = np.stack([(xs - W / 2) * z / self.f, (ys - H / 2) * z / self.f, z], 1).astype(np.float32)
        col = frame[ys, xs].astype(np.float32)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
        col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[ys, xs].astype(np.float32) * 0.9
        m = scene.people_soft(freeze_f)[ys, xs]
        col = col * (1 - 0.55 * m[:, None]) + np.array([255, 235, 170], np.float32) * 0.55 * m[:, None]
        heat_v = scene.heat.at(freeze_f)
        heat = scene.heat.colorize(heat_v, scene.people_soft(freeze_f), gain=1.0)
        col = col + heat[ys, xs] * 0.9
        self.P, self.col = P, np.clip(col, 0, 255)
        self.z0 = z
        cy0 = depth[H // 2 - 150:H // 2 + 150, W // 2 - 200:W // 2 + 200]
        self.c = np.array([0, 0, float(np.median(cy0))], np.float32)
        # Plano do piso, a partir da profundidade deste quadro
        n = np.array(info["floor_normal"], np.float32)
        if n @ np.array([0, -1, 0], np.float32) < 0:
            n = -n
        self.up = n / np.linalg.norm(n)
        fy, fx = np.mgrid[700:1075:6, 200:860:6]
        pf = np.stack([(fx - W / 2) * depth[fy, fx] / self.f, (fy - H / 2) * depth[fy, fx] / self.f, depth[fy, fx]], -1).reshape(-1, 3)
        self.floor_d = float(-np.median(pf @ self.up))
        # Pessoas e olhares no quadro congelado
        self.people = []
        for tid, p in data.by_frame[freeze_f].items():
            if p["head"] is None:
                continue
            x1, y1, x2, y2 = p["head"]
            hx, hy = (x1 + x2) / 2, (y1 + y2) / 2
            hz = float(np.median(depth[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]))
            kp, kc = np.array(p["kp"]), np.array(p["kc"])
            K3 = []
            for (u, v), c in zip(kp, kc):
                if c > 0.4:
                    zz = float(np.clip(depth[int(np.clip(v, 0, H - 1)), int(np.clip(u, 0, W - 1))], hz - 0.4, hz + 0.4))
                    K3.append(self.unproject(u, v, zz))
                else:
                    K3.append(None)
            g = data.gaze_at(tid, freeze_f)
            tgt = None
            if g is not None and g[2] > 0.5 and tid not in data.staff:
                gx, gy = g[0], g[1]
                win = depth[int(max(0, gy - 5)):int(gy + 6), int(max(0, gx - 5)):int(gx + 6)]
                tz = float(np.median(win))
                z_id = data.zone_for(freeze_f, gx, gy)
                tgt = (self.unproject(gx, gy, tz), zone_text(data.zones[z_id]) if z_id >= 0 else None)
            self.people.append({"id": tid, "head": self.unproject(hx, hy, hz), "kp": K3, "kc": kc,
                                "tgt": tgt, "staff": tid in data.staff})

    def unproject(self, u, v, z):
        return np.array([(u - W / 2) * z / self.f, (v - H / 2) * z / self.f, z], np.float32)

    def pose(self, yaw, pitch, dolly):
        view = self.c / np.linalg.norm(self.c)
        right = np.cross(self.up, view)
        right /= np.linalg.norm(right)
        R = cv2.Rodrigues((self.up * math.radians(yaw)).astype(np.float64))[0] @ \
            cv2.Rodrigues((right * math.radians(pitch)).astype(np.float64))[0]
        R = R.astype(np.float32)
        C = self.c + (R @ (-self.c)) * dolly
        return R, C

    def project(self, X, R, C):
        pc = (np.atleast_2d(X) - C) @ R  # = R^T (X - C)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.f * pc[:, 0] / z + W / 2
            v = self.f * pc[:, 1] / z + H / 2
        return u, v, z

    def render(self, yaw, pitch, dolly, shrink=1.0, grid_a=0.0, t=0.0, beams=True, labels_tl=None):
        R, C = self.pose(yaw, pitch, dolly)
        u, v, z = self.project(self.P, R, C)
        ok = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z, col, z0 = u[ok], v[ok], z[ok], self.col[ok], self.z0[ok]
        img = np.zeros((H, W, 3), np.uint8)
        if grid_a > 0:
            self.draw_grid(img, R, C, grid_a)
        fog = np.clip(1.18 - 0.05 * z, 0.5, 1.0)[:, None]
        col = (col * fog).astype(np.uint8)
        size = np.clip(np.round(self.stride * z0 / z * shrink), 1, 6).astype(np.int32)
        ui, vi = u.astype(np.int32), v.astype(np.int32)
        order = np.argsort(z)
        ui, vi, size, col = ui[order], vi[order], size[order], col[order]
        BIG = np.int32(2 ** 30)
        buf = np.full(W * H, BIG, np.int32)
        src_idx = np.arange(len(ui), dtype=np.int32)
        for s in np.unique(size):
            sel = np.nonzero(size == s)[0]
            for dy in range(s):
                for dx in range(s):
                    xx = ui[sel] + dx - s // 2
                    yy = vi[sel] + dy - s // 2
                    good = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                    np.minimum.at(buf, yy[good] * W + xx[good], src_idx[sel[good]])
        filled = buf < BIG
        flat = img.reshape(-1, 3)
        flat[filled] = col[buf[filled]]
        if beams:
            ov = np.zeros((H, W, 3), np.uint8)
            for p in self.people:
                self.draw_person(ov, p, R, C, t, labels_tl)
            img = np.clip(img.astype(np.float32) + ov + glow(ov, 5, 4) * 1.2, 0, 255).astype(np.uint8)
        return img

    def draw_person(self, ov, p, R, C, t, tl):
        col = (130, 130, 130) if p["staff"] else SKEL
        pts = {}
        for i, X in enumerate(p["kp"]):
            if X is not None:
                u, v, z = self.project(X, R, C)
                if z[0] > 0.2:
                    pts[i] = (int(u[0]), int(v[0]))
        for a, b in COCO_EDGES:
            if a in pts and b in pts:
                cv2.line(ov, pts[a], pts[b], col, 2, cv2.LINE_AA)
        u, v, z = self.project(p["head"], R, C)
        if z[0] <= 0.2:
            return
        hc = (float(u[0]), float(v[0]))
        cv2.circle(ov, (int(hc[0]), int(hc[1])), int(max(6, 0.11 * self.f / z[0])), col, 2, cv2.LINE_AA)
        if p["tgt"] is not None:
            X, name = p["tgt"]
            tu, tv, tz = self.project(X, R, C)
            if tz[0] > 0.2:
                draw_beam(ov, hc, (float(tu[0]), float(tv[0])), 1.0, t + p["id"] * 0.37)
                if tl is not None and name:
                    target_label(tl, (float(tu[0]), float(tv[0])), name, 0, 1.0)

    def draw_grid(self, img, R, C, a):
        n = self.up
        view = self.c / np.linalg.norm(self.c)
        e1 = np.cross(n, view)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(e1, n)
        g0 = self.c - (self.c @ n + self.floor_d) * n
        col = tuple(int(c * a) for c in (110, 95, 40))
        ts = np.linspace(-9, 9, 60, dtype=np.float32)
        for i in range(-14, 15):
            for (ea, eb) in ((e1, e2), (e2, e1)):
                X = g0 + ea * (i * 0.6) + eb * ts[:, None]
                u, v, z = self.project(X, R, C)
                ok = z > 0.3
                pts = np.stack([u, v], 1)
                for k in range(len(ts) - 1):
                    if ok[k] and ok[k + 1] and abs(u[k]) < 4 * W and abs(v[k]) < 4 * H:
                        cv2.line(img, (int(u[k]), int(v[k])), (int(u[k + 1]), int(v[k + 1])), col, 1, cv2.LINE_AA)


CAM_KEYS = [(0.0, (0.0, 0.0, 1.0)), (0.33, (-16.0, 10.0, 0.86)), (0.68, (13.0, -5.0, 0.90)), (1.0, (0.0, 0.0, 1.0))]


def cam_path(u):
    """Catmull-Rom pelos pontos de vista (giro, inclinação, aproximação)."""
    ts = [k[0] for k in CAM_KEYS]
    P = [np.array(k[1], np.float64) for k in CAM_KEYS]
    u = clamp01(u)
    i = min(len(ts) - 2, max(j for j in range(len(ts) - 1) if ts[j] <= u))
    s = (u - ts[i]) / (ts[i + 1] - ts[i])
    p0, p1, p2 = P[max(0, i - 1)], P[i], P[i + 1]
    p3 = P[min(len(P) - 1, i + 2)]
    v = 0.5 * (2 * p1 + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3)
    return float(v[0]), float(v[1]), float(v[2])


# ---------------------------------------------------------------- linha do tempo

class Timeline:
    def __init__(self, freeze_src, src_end, end_dur=3.5, report_dur=7.5):
        self.freeze_src = freeze_src
        self.src_end = src_end
        self.seg = []
        t = 0.0
        for name, dur in [("raw", 5.2), ("scan", 2.0), ("rewind", 0.8), ("live", freeze_src), ("freeze3d", 10.0),
                          ("ff", 6.0), ("report", report_dur), ("end", end_dur)]:
            if dur <= 0:  # cena removida (ex.: sem tela final)
                continue
            self.seg.append((name, t, t + dur))
            t += dur
        self.total = t

    def at(self, t):
        for name, t0, t1 in self.seg:
            if t < t1 or (name, t0, t1) == self.seg[-1]:
                lt = t - t0
                if name in ("raw",):
                    src = lt
                elif name == "scan":
                    src = 5.2 + lt
                elif name == "rewind":
                    src = 7.2 * (1 - ease_in_out(lt / (t1 - t0)))
                elif name == "live":
                    src = lt
                elif name == "freeze3d":
                    src = self.freeze_src
                elif name == "ff":
                    src = self.freeze_src + (self.src_end - self.freeze_src) * lt / (t1 - t0)
                else:
                    src = self.src_end
                return name, lt, t1 - t0, src
        raise ValueError(t)

    def start(self, name):
        return next(t0 for n, t0, _ in self.seg if n == name)


# ---------------------------------------------------------------- quadros

class Renderer:
    def __init__(self, data_dir, freeze_src):
        self.d = Data(data_dir)
        self.S = Scene(self.d, self.d.meta["video"])
        fps = self.d.meta["fps"]
        self.fps = fps
        self.src_end = (self.d.n_frames - 1) / fps
        if CREDITS["watermark"]:
            # Sem tela final: o relatório fecha o vídeo, com fade para preto
            self.tl = Timeline(freeze_src, self.src_end, end_dur=0, report_dur=8.5)
        else:
            self.tl = Timeline(freeze_src, self.src_end, end_dur=5.0 if CREDITS["author"] else 3.5)
        self.freeze_f = int(round(freeze_src * fps))
        self.cloud = None
        self.report_bg = None
        self.freeze_live = None

    def frame(self, t):
        name, lt, dur, src = self.tl.at(t)
        f = int(round(src * self.fps))
        img = getattr(self, "r_" + name)(t, lt, dur, f, src)
        if CREDITS["watermark"]:
            img = watermark(img)
            tail = self.tl.total - t
            if tail < 0.8:  # fade para preto no fim
                img = (img.astype(np.float32) * max(0.0, tail / 0.8)).astype(np.uint8)
        return img

    # Câmera crua, com cara de gravação
    def r_raw(self, t, lt, dur, f, src):
        img = self.S.src(f).copy()
        tl = TextLayer()
        self.cctv_hud(img, tl, t, src)
        # Gradiente para os títulos
        g = np.linspace(0.62, 0.0, 900, dtype=np.float32)[None, :, None]
        img[:, :900] = (img[:, :900].astype(np.float32) * (1 - g)).astype(np.uint8)
        a1 = fade(lt, 0.25, 2.65, 0.5, 0.35)
        a2 = fade(lt, 2.85, 5.2, 0.5, 0.0)
        for a, s, t0 in ((a1, TX["raw1"], 0.25), (a2, TX["raw2"], 2.85)):
            if a > 0:
                dy = (1 - ease_out((lt - t0) / 0.7)) * 24
                tl.text((90, 470 + dy), s, font("bold", 76), RGB_WHITE, a=a, shadow=True)
        return tl.composite(img)

    def cctv_hud(self, img, tl, t, src):
        tl.text((44, 36), "CAM 01", font("monob", 26), RGB_WHITE, shadow=True)
        if int(t * 2) % 2 == 0:
            cv2.circle(img, (W - 150, 52), 10, (40, 40, 235), -1, cv2.LINE_AA)
        tl.text((W - 128, 36), "REC", font("monob", 26), RGB_WHITE, shadow=True)
        tl.text((W - 44, H - 64), f"00:00:{int(src):02d}", font("mono", 24), RGB_WHITE, anchor="ra", shadow=True)

    # Varredura: acima da linha, "modo IA"; abaixo, câmera crua
    def r_scan(self, t, lt, dur, f, src):
        raw = self.S.src(f).copy()
        live, tl, n_now = self.S.live(f, t, {"labels": False})
        y = int(ease_in_out(lt / (dur * 0.85)) * (H + 40))
        alpha = np.clip((y - np.arange(H, dtype=np.float32)) / 30.0, 0, 1)[:, None, None]
        img = (live * alpha + raw * (1 - alpha)).astype(np.uint8)
        line = np.zeros((H, W, 3), np.uint8)
        if 0 <= y < H:
            cv2.line(line, (0, y), (W, y), (255, 245, 170), 3)
            img = np.clip(img.astype(np.float32) + line + glow(line, 6, 4) * 2.2, 0, 255).astype(np.uint8)
            tl2 = TextLayer()
            tl2.text((W - 60, y - 34), TX["scan"], font("monob", 16), RGB_CYAN, anchor="ra", track=2)
            img = tl2.composite(img)
        # Só mostra o texto da camada ao vivo acima da linha
        arr = np.asarray(tl.im).copy()
        arr[max(0, y):, :, 3] = 0
        tl.im = Image.fromarray(arr)
        hud_live(tl, img, t, src) if y > 140 else None
        a = fade(lt, 0.2, dur, 0.4, 0.0)
        tl.text((90, 470), TX["scan_title"], font("bold", 76), RGB_WHITE, a=a, shadow=True)
        return tl.composite(img)

    # Rebobina até o início para mostrar a mesma cena, agora com IA
    def r_rewind(self, t, lt, dur, f, src):
        img, tl, n_now = self.S.live(f, t, {"labels": False, "tags": False})
        # Linhas de "fita rebobinando"
        rng = np.random.default_rng(int(t * 1000))
        for _ in range(5):
            y = int(rng.integers(140, H - 140))
            img[y:y + 3] = np.clip(img[y:y + 3].astype(np.int16) + 70, 0, 255).astype(np.uint8)
        hud_live(tl, img, t, src, mode="ff", badge="◀◀ 9×")
        tl.text((W / 2, 150), TX["rewind"], font("monob", 22), RGB_CYAN, anchor="ma", track=4)
        return tl.composite(img)

    def r_live(self, t, lt, dur, f, src):
        img, tl, n_now = self.S.live(f, t, {})
        hud_live(tl, img, t, src)
        zone_panel(tl, img, self.d, f, n_now)
        half = dur / 2
        caption(tl, lt, 0.2, half - 0.1, TX["live1"], TX["live1_sub"])
        caption(tl, lt, half + 0.1, dur - 0.15, TX["live2"], TX["live2_sub"])
        return tl.composite(img)

    def ensure_cloud(self):
        if self.cloud is None:
            self.cloud = Cloud(self.d, self.S, self.freeze_f)
            img, tl, n_now = self.S.live(self.freeze_f, 0.0, {})
            self.freeze_live = img

    def r_freeze3d(self, t, lt, dur, f, src):
        self.ensure_cloud()
        tl = TextLayer()
        # Trajetória da câmera virtual: órbita moderada (a profundidade vem de uma
        # única imagem; giros grandes mostram as superfícies esticadas)
        yaw, pitch, dolly = cam_path(ease_in_out((lt - 1.4) / (dur - 2.7)))
        shrink = 1.0 - 0.32 * ease_in_out((lt - 0.7) / 0.9) + 0.32 * ease_in_out((lt - (dur - 1.3)) / 0.9)
        grid_a = ease_in_out((lt - 1.3) / 1.0) * (1 - ease_in_out((lt - (dur - 1.6)) / 0.8))
        cloud = self.cloud.render(yaw, pitch, dolly, shrink=shrink, grid_a=grid_a, t=t,
                                  beams=True, labels_tl=tl if 1.6 < lt < dur - 1.4 else None)
        mix = ease_in_out((lt - 0.6) / 0.9) * (1 - ease_in_out((lt - (dur - 1.1)) / 0.9))
        img = (self.freeze_live.astype(np.float32) * (1 - mix) + cloud.astype(np.float32) * mix)
        flash = max(0.0, 1 - lt / 0.35) * 0.55
        img = np.clip(img + 255 * flash, 0, 255).astype(np.uint8)
        hud_live(tl, img, t, src, mode="3d")
        a = fade(lt, 0.15, 2.4, 0.2, 0.4)
        tl.text((W / 2, 150), TX["frozen"], font("monob", 22), RGB_CYAN, a=a, anchor="ma", track=6)
        caption(tl, lt, 1.6, dur - 1.2, TX["3d"], TX["3d_sub"])
        return tl.composite(img)

    def r_ff(self, t, lt, dur, f, src):
        img, tl, n_now = self.S.live(f, t, {"labels": False, "tags": False})
        speed = (self.src_end - self.tl.freeze_src) / dur
        hud_live(tl, img, t, src, mode="ff", badge=f"▶▶ {int(round(speed))}×")
        zone_panel(tl, img, self.d, f, n_now)
        caption(tl, lt, 0.2, dur - 0.1, TX["ff"].format(s=int(round(self.src_end))), TX["ff_sub"])
        return tl.composite(img)

    def r_report(self, t, lt, dur, f, src):
        if self.report_bg is None:
            base = self.S.analysis_base(self.S.plate, darkness=0.30)
            v = self.S.heat.at(self.d.n_frames - 1)
            base += self.S.heat.colorize(v, None, gain=1.15)
            self.report_bg = np.clip(base, 0, 255).astype(np.uint8)
        # A cena encolhe para a esquerda
        k = ease_in_out(lt / 0.9)
        sc = 1 - 0.36 * k
        sw, sh = int(W * sc), int(H * sc)
        x0, y0 = int(70 * k), int(170 * k)
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (14, 10, 6)
        scene = cv2.resize(self.report_bg, (sw, sh), interpolation=cv2.INTER_AREA)
        img[y0:y0 + sh, x0:x0 + sw] = scene
        cv2.rectangle(img, (x0, y0), (x0 + sw - 1, y0 + sh - 1), (90, 75, 30), 1)
        tl = TextLayer()
        vals = self.d.cum_zone[-1]
        total = vals.sum()
        order = [int(i) for i in np.argsort(-vals) if vals[i] > 0][:6]
        a_t = fade(lt, 0.6, dur, 0.5, 0.0)
        tl.text((70, 52), TX["report"], font("bold", 54), RGB_WHITE, a=a_t)
        tl.text((72, 124), TX["report_sub"].format(s=int(round(self.src_end))),
                font("mono", 18), RGB_DIM, a=a_t, track=1)
        # Marcadores numerados nas três zonas mais olhadas
        for r, zi in enumerate(order[:3]):
            a = fade(lt, 1.6 + r * 0.25, dur, 0.4, 0.0)
            if a <= 0:
                continue
            cx, cy = self.d.centroids[zi]
            px, py = x0 + cx * sc, y0 + cy * sc
            cv2.circle(img, (int(px), int(py)), int(22 * ease_out((lt - 1.6 - r * 0.25) / 0.4)), AMBER, -1, cv2.LINE_AA)
            tl.text((px, py), str(r + 1), font("num", 30), (20, 12, 0), a=a, anchor="mm")
        # Ranking à direita
        px0 = 1340
        tl.text((px0, 196), TX["ranking"], font("monob", 18), RGB_CYAN, a=a_t, track=3)
        for r, zi in enumerate(order):
            a = fade(lt, 1.0 + r * 0.15, dur, 0.4, 0.0)
            if a <= 0:
                continue
            yy = 250 + r * 92
            share = vals[zi] / total
            grow = ease_out((lt - 1.0 - r * 0.15) / 1.1)
            tl.text((px0, yy), f"{r + 1}", font("num", 40), RGB_AMBER if r < 3 else RGB_DIM, a=a)
            tl.text((px0 + 46, yy + 2), zone_text(self.d.zones[zi], "name"), font("demi", 26), RGB_WHITE, a=a)
            tl.text((W - 70, yy - 2), f"{int(round(share * 100 * grow))}%", font("num", 44), RGB_AMBER, a=a, anchor="ra")
            bw = W - 70 - (px0 + 46)
            cv2.rectangle(img, (px0 + 46, yy + 48), (px0 + 46 + bw, yy + 54), (60, 48, 22), -1)
            cv2.rectangle(img, (px0 + 46, yy + 48), (px0 + 46 + int(bw * share * grow), yy + 54), AMBER, -1)
            tl.text((px0 + 46, yy + 62), TX["rank_row"].format(s=fmt_dec(vals[zi]), n=self.d.viewers.get(zi, 0)),
                    font("mono", 14), RGB_DIM, a=a)
        # Indicadores
        kp = ease_out((lt - 2.2) / 1.6)
        a_k = fade(lt, 2.2, dur, 0.5, 0.0)
        kpis = [(fmt_int(self.d.cum_count[-1] * kp), TX["kpi_gazes"]),
                (fmt_dec(total * kp), TX["kpi_secs"]),
                (str(int((vals > 0).sum())), TX["kpi_zones"])]
        for i, (v, lab) in enumerate(kpis):
            xx = 70 + i * 420
            tl.text((xx, 905), v, font("num", 84), RGB_WHITE, a=a_k)
            tl.text((xx + 4, 1000), lab.upper(), font("mono", 16), RGB_CYAN, a=a_k, track=2)
        # Notas de método (o que entra e o que não entra na conta)
        a_n = fade(lt, 3.0, dur, 0.6, 0.0)
        for i, s in enumerate(TX["notes"]):
            tl.text((px0, 880 + i * 30), "·  " + s, font("mono", 15), RGB_DIM, a=a_n)
        return tl.composite(img)

    def r_end(self, t, lt, dur, f, src):
        if CREDITS["author"]:
            return self.r_end_personal(lt, dur)
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (10, 7, 4)
        tl = TextLayer()
        a = fade(lt, 0.2, dur, 0.6, 0.6)
        track = int(34 - 20 * ease_out(lt / 1.6))
        tl.text((W / 2, 440), "BRANDPOSE", font("heavy", 110), RGB_WHITE, a=a, anchor="ma", track=track)
        line = np.zeros((H, W, 3), np.uint8)
        half = int(330 * ease_out((lt - 0.3) / 1.0))
        cv2.line(line, (W // 2 - half, 600), (W // 2 + half, 600), tuple(int(c * a) for c in AMBER), 3)
        img = np.clip(img.astype(np.float32) + line + glow(line, 6, 4) * 2.0, 0, 255).astype(np.uint8)
        tl.text((W / 2, 630), TX["end_tagline"], font("medium", 40), (220, 230, 238),
                a=fade(lt, 0.6, dur, 0.6, 0.6), anchor="ma")
        tl.text((W / 2, H - 70), TX["end_credits"],
                font("mono", 15), RGB_DIM, a=fade(lt, 0.9, dur, 0.6, 0.6), anchor="ma", track=1)
        return tl.composite(img)

    # Encerramento da versão "projeto pessoal": sem marca, com o autor em destaque
    def r_end_personal(self, lt, dur):
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (10, 7, 4)
        tl = TextLayer()
        tl.text((W / 2, 340), CREDITS["tagline"], font("medium", 38), (205, 225, 235),
                a=fade(lt, 0.2, dur, 0.6, 0.6), anchor="ma")
        line = np.zeros((H, W, 3), np.uint8)
        half = int(300 * ease_out((lt - 0.5) / 1.0))
        a_line = fade(lt, 0.5, dur, 0.3, 0.6)
        cv2.line(line, (W // 2 - half, 440), (W // 2 + half, 440), tuple(int(c * a_line) for c in AMBER), 3)
        img = np.clip(img.astype(np.float32) + line + glow(line, 6, 4) * 2.0, 0, 255).astype(np.uint8)
        tl.text((W / 2, 490), TX["created_by"], font("mono", 22), RGB_CYAN, a=fade(lt, 0.8, dur, 0.5, 0.6),
                anchor="ma", track=8)
        track = int(30 - 18 * ease_out((lt - 1.0) / 1.6))
        tl.text((W / 2, 540), CREDITS["author"].upper(), font("heavy", 100), RGB_WHITE,
                a=fade(lt, 1.0, dur, 0.6, 0.6), anchor="ma", track=track)
        if CREDITS["link"]:
            tl.text((W / 2, 700), CREDITS["link"], font("medium", 40), (175, 215, 240),
                    a=fade(lt, 1.6, dur, 0.6, 0.6), anchor="ma")
        return tl.composite(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out")
    ap.add_argument("--freeze", type=float, default=19.0, help="instante do vídeo de origem a congelar (s)")
    ap.add_argument("--stills", default="", help="tempos de saída (s) para renderizar como JPG")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--author", default="", help="versão 'projeto pessoal': sem marca, com o autor no final")
    ap.add_argument("--author-link", default="")
    ap.add_argument("--credits", choices=["endcard", "watermark"], default="endcard",
                    help="com --author: tela final com o autor, ou só marca d'água no rodapé")
    ap.add_argument("--lang", choices=sorted(TEXTS), default="pt", help="idioma dos textos na tela")
    args = ap.parse_args()

    TX.update(TEXTS[args.lang])
    if "tagline" in TX:
        CREDITS["tagline"] = TX["tagline"]
    if args.author:
        CREDITS.update(title=TX["personal_title"], sub=TX["personal_sub"],
                       author=args.author, link=args.author_link or None,
                       watermark=args.credits == "watermark")
    r = Renderer(args.data_dir, args.freeze)
    timeline = {"total": r.tl.total, "segments": r.tl.seg, "freeze_src": args.freeze}
    print(json.dumps(timeline))
    if args.stills:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for s in args.stills.split(","):
            t = float(s)
            cv2.imwrite(str(out / f"still_{t:05.1f}.jpg"), r.frame(t), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    out = Path(args.out)
    (out.parent / "timeline.json").write_text(json.dumps(timeline, indent=1))
    t_end = args.end or r.tl.total
    n = int(round((t_end - args.start) * FPS))
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    for i in range(n):
        t = args.start + i / FPS
        ff.stdin.write(r.frame(t).tobytes())
        if i % 60 == 0:
            print(f"{t:.1f}s / {t_end:.1f}s", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
