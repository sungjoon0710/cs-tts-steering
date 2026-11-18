"""Praat-style views (waveform, spectrogram + pitch, word tier) of pre- vs post-alignment clips.

Spectrogram and pitch use Praat's own algorithms through Parselmouth (Jadoul et al. 2018).
For each --folders entry, draws <run>/<folder>/pre.TextGrid over the old clip and, if present,
post.TextGrid over <run>/<folder>/*_aligned.wav, side by side. Writes <run>/figures/<folder>.png.
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth

PITCH = (50, 400)       # Hz, Praat's default pitch display range
SPEC_MAX = 5000         # Hz, Praat's default spectrogram view range
DYN_RANGE = 50          # dB, Praat's default dynamic range


def read_textgrid(path):
    txt = Path(path).read_text()
    return [(float(a), float(b), t) for a, b, t in
            re.findall(r'xmin = ([\d.e-]+)\s+xmax = ([\d.e-]+)\s+text = "(.*)"', txt)]


def panel(fig, spec, wav, tg, title):
    snd = parselmouth.Sound(str(wav)).convert_to_mono()
    gs = spec.subgridspec(3, 1, height_ratios=[1.2, 1.6, 0.7], hspace=0.05)
    ax_w, ax_s, ax_t = (fig.add_subplot(g) for g in gs)
    t = snd.xs()
    ax_w.plot(t, snd.values[0], color="black", lw=0.4)
    ax_w.set_title(title, fontsize=10, loc="left")

    sg = snd.to_spectrogram(window_length=0.005, maximum_frequency=SPEC_MAX)
    db = 10 * np.log10(np.maximum(sg.values, 1e-12))
    ax_s.pcolormesh(sg.x_grid(), sg.y_grid(), db, cmap="Greys", vmin=db.max() - DYN_RANGE, vmax=db.max(),
                    shading="flat", rasterized=True)
    ax_s.set_ylim(0, SPEC_MAX)
    ax_p = ax_s.twinx()
    pitch = snd.to_pitch(pitch_floor=PITCH[0], pitch_ceiling=PITCH[1])
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    ax_p.plot(pitch.xs(), f0, color="tab:blue", lw=1.5)
    ax_p.set_ylim(*PITCH)
    ax_p.set_ylabel("F0 (Hz)", fontsize=8, color="tab:blue")

    for a, b, w in read_textgrid(tg):
        unmatched = w == "*"
        ax_t.axvspan(a, b, color="#d9d9a0" if unmatched else "white", ec="black", lw=0.8)
        if w and (b - a) > (0.3 if unmatched else 0.06):  # skip labels too narrow to read
            ax_t.text((a + b) / 2, 0.5, "‹unmatched›" if unmatched else w, ha="center", va="center",
                      fontsize=8, rotation=90 if b - a < 0.25 and not unmatched else 0, family="serif")
    for ax in (ax_w, ax_s, ax_t):
        ax.set_xlim(0, snd.duration)
    for ax in (ax_w, ax_s):
        ax.set_xticks([])
    ax_w.set_yticks([])
    ax_s.set_ylabel("Hz", fontsize=8)
    ax_t.set_yticks([])
    ax_t.set_xlabel("time (s)", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="scripts/realign.py output dir")
    ap.add_argument("--items", type=Path, required=True, help="Listening_Examples_100_PT dir (old clips)")
    ap.add_argument("--folders", nargs="+", required=True, help="folder-name prefixes, e.g. 03_train_1440_3")
    args = ap.parse_args()
    (args.run / "figures").mkdir(exist_ok=True)
    for prefix in args.folders:
        d = next(p for p in sorted(args.run.iterdir()) if p.is_dir() and p.name.startswith(prefix))
        old = next((args.items / d.name).glob("01_Natural_*.wav"))
        views = [(old, d / "pre.TextGrid", f"Pre-alignment: {old.name}")]
        post = sorted(d.glob("*_aligned.wav"))
        if post and (d / "post.TextGrid").exists():
            views.append((post[0], d / "post.TextGrid", f"Post-alignment: {post[0].name}"))
        fig = plt.figure(figsize=(8 * len(views), 5))
        gs = fig.add_gridspec(1, len(views), wspace=0.15)
        for g, (wav, tg, title) in zip(gs, views):
            panel(fig, g, wav, tg, title)
        fig.suptitle(d.name, fontsize=11)
        out = args.run / "figures" / f"{d.name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
