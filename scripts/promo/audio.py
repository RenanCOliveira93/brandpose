"""Trilha sonora sintetizada (original, sem música de terceiros) para o vídeo promocional.

Lê timeline.json (gerado pelo render_promo.py) e sincroniza os eventos com as cenas.

Uso: python audio.py TIMELINE.json OUT.wav
"""

import json
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
rng = np.random.default_rng(7)


def T(d):
    return np.arange(int(d * SR)) / SR


def lp(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, fc, "low", fs=SR, output="sos"), x, axis=0)


def hp(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, fc, "high", fs=SR, output="sos"), x, axis=0)


def env(n, a, r, curve=1.0):
    """Envelope com ataque e liberação lineares (em segundos), sem cliques."""
    e = np.ones(n)
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na) ** curve
    e[-nr:] *= np.linspace(1, 0, nr) ** curve
    return e


def place(buf, x, t0, gain=1.0):
    i0 = int(t0 * SR)
    if i0 >= len(buf):
        return
    x = x[: len(buf) - i0]
    if x.ndim == 1 and buf.ndim == 2:
        x = np.stack([x, x], 1)
    buf[i0:i0 + len(x)] += x * gain


def voice(f, d, harmonics=8, tilt=1.6, detune_cents=4):
    t = T(d)
    out = np.zeros((len(t), 2))
    for ch, cents in enumerate((-detune_cents, detune_cents)):
        ff = f * 2 ** (cents / 1200)
        for k in range(1, harmonics + 1):
            out[:, ch] += np.sin(2 * np.pi * ff * k * t + rng.uniform(0, 6.28)) / k ** tilt
    return out


def pad(freqs, d, a=1.5, r=1.5, cutoff=1600):
    x = sum(voice(f, d) for f in freqs) / len(freqs)
    lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.23 * T(d))[:, None]
    return lp(x * lfo, cutoff) * env(len(x), a, r)[:, None]


def kick(gain=1.0):
    t = T(0.45)
    f = 45 + 80 * np.exp(-t / 0.045)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / 0.13) * env(len(t), 0.002, 0.05) * gain


def hat():
    t = T(0.05)
    return hp(rng.standard_normal(len(t)), 7000) * np.exp(-t / 0.012) * env(len(t), 0.0015, 0.01)


def pluck(f, d=1.6):
    t = T(d)
    x = sum(np.sin(2 * np.pi * f * k * t) * np.exp(-t * (2.2 + 1.8 * k)) / k for k in range(1, 6))
    return x * env(len(t), 0.004, 0.2)


def blip():
    a = np.sin(2 * np.pi * 1318.5 * T(0.07)) * env(int(0.07 * SR), 0.003, 0.05)
    b = np.sin(2 * np.pi * 1975.5 * T(0.09)) * env(int(0.09 * SR), 0.003, 0.07)
    out = np.zeros(int(0.16 * SR))
    out[:len(a)] += a
    out[int(0.05 * SR):int(0.05 * SR) + len(b)] += b
    return out


def riser(d, f0=300, f1=6000):
    """Ruído com filtro abrindo (cruzando versões filtradas) + tom subindo."""
    n = int(d * SR)
    noise = rng.standard_normal((n, 2))
    cuts = np.geomspace(f0, f1, 5)
    layers = [lp(noise, c) for c in cuts]
    u = np.linspace(0, len(cuts) - 1, n)
    out = np.zeros((n, 2))
    for i, layer in enumerate(layers):
        w = np.clip(1 - np.abs(u - i), 0, 1)[:, None]
        out += layer * w
    tone_f = np.geomspace(180, 900, n)
    tone = np.sin(2 * np.pi * np.cumsum(tone_f) / SR)[:, None] * 0.25
    return (out * 0.5 + tone) * (np.linspace(0, 1, n) ** 2 * env(n, 0.001, 0.01))[:, None]


def boom(d=2.5):
    t = T(d)
    f = 38 + 30 * np.exp(-t / 0.25)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.7)
    hit = lp(rng.standard_normal(len(t)), 260) * np.exp(-t / 0.12) * 0.8
    return (sub + hit) * env(len(t), 0.003, 0.3)


def chime(freqs, d=2.5):
    t = T(d)
    return sum(np.sin(2 * np.pi * f * t) * np.exp(-t / (0.9 * 440 / f + 0.4)) for f in freqs) / len(freqs) * env(len(t), 0.003, 0.4)


def rewind(d):
    t = T(d)
    f = 1700 * (240 / 1700) ** (t / d)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.55 + 0.45 * np.sin(2 * np.pi * 28 * t))
    nz = lp(rng.standard_normal(len(t)), 3500) * 0.3
    return lp(s + nz, 4500) * env(len(t), 0.03, 0.12)


def reverb(x, seconds=2.4):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-t / (seconds / 6.9))[:, None]
    ir = lp(ir, 5000)
    ir /= np.sqrt((ir ** 2).sum(0))
    return np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)


def main():
    tl = json.loads(open(sys.argv[1]).read())
    seg = {name: (t0, t1) for name, t0, t1 in tl["segments"]}
    total = tl["total"]
    n = int(total * SR) + SR
    dry = np.zeros((n, 2))
    wet = np.zeros((n, 2))
    duck = np.ones(n)

    A2, C3, E3, G3, A3, B3, C4, E4, G4 = 110.0, 130.81, 164.81, 196.0, 220.0, 246.94, 261.63, 329.63, 392.0
    F2, F3, D4 = 87.31, 174.61, 293.66
    C2, G2 = 65.41, 98.0

    # Abertura: zumbido de câmera e ruído de ambiente
    r0, r1 = seg["raw"]
    hum = sum(np.sin(2 * np.pi * f * T(r1 - r0 + 0.6)) / k for k, f in enumerate((60, 120, 180), 1))
    place(dry, (hum * 0.6 + lp(rng.standard_normal(len(hum)), 1800) * 0.25) * env(len(hum), 0.6, 0.8), r0, 0.05)

    # Camas harmônicas por cena
    s0 = seg["scan"][0]
    f0_, f1_ = seg["freeze3d"]
    ff0, ff1 = seg["ff"]
    rp0 = seg["report"][0]
    place(dry, pad([A2, E3, A3, B3, C4, E4], f0_ - s0 + 0.8, a=2.0, r=0.8), s0 - 0.3, 0.30)
    place(dry, pad([F2, C3, F3, A3, E4, G4], f1_ - f0_ + 0.9, a=0.6, r=0.9, cutoff=2200), f0_ - 0.1, 0.30)
    place(dry, pad([A2, E3, A3, C4, E4, B3 * 2], ff1 - ff0 + 0.8, a=0.5, r=0.8, cutoff=2600), ff0 - 0.2, 0.28)
    place(dry, pad([C2, G2, C3, E3, G3, D4, E4], total - rp0 + 0.2, a=0.8, r=2.2, cutoff=2000), rp0 - 0.2, 0.30)
    place(wet, pad([A3, E4, B3 * 2], f1_ - f0_, a=2.0, r=2.0, cutoff=5000), f0_, 0.10)

    # Varredura e "Até agora."
    sc0, sc1 = seg["scan"]
    place(dry, riser(sc1 - sc0, 300, 7000), sc0, 0.10)
    place(wet, blip(), sc0 + 0.2, 0.25)
    place(dry, boom(1.8), sc0 + 0.2, 0.35)

    # Rebobinar
    rw0, rw1 = seg["rewind"]
    place(dry, rewind(rw1 - rw0), rw0, 0.22)

    # Pulso durante a análise ao vivo (100 bpm) e no avanço rápido (150 bpm)
    for a_, b_, period, kg in ((seg["live"][0], seg["live"][1], 0.6, 0.55), (ff0, ff1, 0.4, 0.5)):
        t = a_
        while t < b_ - 0.1:
            place(dry, kick(), t, kg)
            i0 = int(t * SR)
            dl = int(0.35 * SR)
            duck[i0:i0 + dl] = np.minimum(duck[i0:i0 + dl], 1 - 0.3 * np.exp(-np.arange(min(dl, n - i0)) / (0.08 * SR)))
            place(dry, hat(), t + period / 2, 0.05)
            t += period

    # Congelamento: subida, impacto e sino
    place(dry, riser(1.2, 200, 5000), f0_ - 1.2, 0.12)
    place(dry, boom(3.0), f0_, 0.55)
    place(wet, chime([1760, 2637, 3520], 3.0), f0_ + 0.02, 0.20)
    # Brilhos durante a órbita 3D
    notes = [880.0, 1046.5, 1318.5, 1568.0, 1760.0]
    t = f0_ + 1.2
    while t < f1_ - 1.0:
        place(wet, pluck(rng.choice(notes), 1.2), t, 0.06)
        t += rng.uniform(0.28, 0.5)
    # Vento suave acompanhando o movimento da câmera
    orb = T(f1_ - f0_ - 2.6)
    wind = lp(rng.standard_normal((len(orb), 2)), 700) * np.sin(np.pi * orb / orb[-1])[:, None] ** 2
    place(dry, wind, f0_ + 1.4, 0.10)

    # Avanço rápido: tom subindo
    fft = T(ff1 - ff0)
    glide = np.sin(2 * np.pi * np.cumsum(np.geomspace(220, 440, len(fft))) / SR) * env(len(fft), 0.3, 0.3)
    place(dry, lp(glide, 2000), ff0, 0.05)

    # Legendas e itens do ranking
    for tc in (0.25, 2.85, seg["live"][0] + 0.2, (seg["live"][0] + seg["live"][1]) / 2 + 0.1, f0_ + 1.6, ff0 + 0.2, rp0 + 0.6):
        place(wet, blip(), tc, 0.12)
    for r, f in enumerate([E4, G4, A3 * 2, C4 * 2, D4 * 2, E4 * 2]):
        place(wet, pluck(f), rp0 + 1.0 + r * 0.15, 0.22)
    place(dry, riser(0.8, 400, 4000), rp0 + 1.4, 0.05)

    # Encerramento (só quando existe a tela final)
    if "end" in seg:
        e0 = seg["end"][0]
        place(dry, boom(2.5), e0 + 0.2, 0.35)
        place(wet, chime([523.25, 659.26, 783.99, 1046.5], 3.0), e0 + 0.2, 0.30)

    mix = dry * duck[:, None] + reverb(wet) * 1.0 + wet * 0.5
    mix = mix[: int(total * SR)]
    mix *= env(len(mix), 0.05, 1.2)[:, None]
    mix = np.tanh(1.3 * mix) / np.tanh(1.3)
    mix *= 0.89 / np.abs(mix).max()  # pico em -1 dBFS
    jumps = np.abs(np.diff(mix, axis=0)).max()
    wavfile.write(sys.argv[2], SR, (mix * 32767).astype(np.int16))
    rms = np.sqrt((mix ** 2).mean())
    print(json.dumps({"seconds": round(len(mix) / SR, 2), "peak_dbfs": round(20 * np.log10(np.abs(mix).max()), 2),
                      "rms_dbfs": round(20 * np.log10(rms), 2), "max_sample_jump": round(float(jumps), 3)}))


if __name__ == "__main__":
    main()
