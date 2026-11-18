"""Figures and LaTeX tables for the n-shot style prompting report. Input: nshot_per_clip.csv (IDs + numbers only)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("nshot_per_clip.csv")
d["capped"] = d.hit_max.astype(str).eq("True")
SPK = [("MAYA", "Maya"), ("INDRAVARDHAN", "Indravardhan")]
GEN = ["base", "nat1", "nat3", "nat5", "syn3"]
LBL = {"natural": "natural", "base": "base\n(0 shots)", "nat1": "nat-1", "nat3": "nat-3", "nat5": "nat-5", "syn3": "syn-3"}
BLUE, ORANGE, AQUA, GRAY, INK, MUT = '#2a78d6', '#eb6834', '#1baf7a', '#b4b2a9', '#1f1f1e', '#6b6a64'
plt.rcParams.update({'font.size': 10, 'font.family': ['Helvetica'], 'axes.edgecolor': MUT, 'axes.labelcolor': INK,
                     'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white'})


def grid(ax):
    ax.grid(axis='y', color='#ebeae6'); ax.set_axisbelow(True)


# Fig 1: runaway generations (no EOS before the token cap)
fig, ax = plt.subplots(figsize=(6.6, 3.0)); x = np.arange(len(GEN)); w = .36
for k, ((s, name), c) in enumerate(zip(SPK, (BLUE, ORANGE))):
    v = [int(d[(d.speaker == s) & (d.condition == g)].capped.sum()) for g in GEN]
    bars = ax.bar(x + (k - .5) * (w + .02), v, w, color=c, label=name)
    for b, n in zip(bars, v):
        ax.annotate(f'{n}', (b.get_x() + b.get_width() / 2, n), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9, color=INK)
ax.set_xticks(x, [LBL[g] for g in GEN]); ax.set_ylim(0, 15); ax.set_ylabel('Clips that hit the token cap\n(of 15 per speaker)')
ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper left'); grid(ax)
fig.tight_layout(); fig.savefig('fig1_runaway.pdf')

# Fig 2: speaker similarity per clip, coloured by capped / stopped on its own
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
rng = np.random.default_rng(0)
for ax, (s, name) in zip(axes, SPK):
    conds = ["natural"] + GEN
    for i, g in enumerate(conds):
        q = d[(d.speaker == s) & (d.condition == g)]
        for capped, c in ((False, BLUE), (True, ORANGE)):
            r = q[q.capped == capped]
            ax.scatter(i + rng.uniform(-.18, .18, len(r)), r.ssim, s=14, color=c if g != "natural" else GRAY,
                       alpha=.85, lw=0, label=None)
    ax.set_xticks(range(len(conds)), [LBL[g] for g in conds], fontsize=8); ax.set_title(name, fontsize=10, color=INK)
    ax.set_ylim(0, 1.02); grid(ax)
axes[0].set_ylabel('Speaker similarity to the\nspeaker\'s natural clips (S-SIM)')
h = [plt.Line2D([], [], marker='o', ls='', color=c, ms=5) for c in (GRAY, BLUE, ORANGE)]
fig.legend(h, ['natural (leave-one-out ceiling)', 'stopped on its own', 'hit the token cap'], frameon=False, fontsize=9,
           ncol=3, loc='lower center', bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig('fig2_ssim.pdf')

# Fig 3: F0 variability, all clips (bars) vs clips that stopped on their own (dots)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, (s, name) in zip(axes, SPK):
    nat = d[(d.speaker == s) & (d.condition == "natural")].f0_std_st.mean()
    allm = [d[(d.speaker == s) & (d.condition == g)].f0_std_st.mean() for g in GEN]
    cln = [d[(d.speaker == s) & (d.condition == g) & ~d.capped] for g in GEN]
    ax.bar(range(len(GEN)), allm, .6, color='#cfe0f5', label='all 15 clips')
    ax.scatter(range(len(GEN)), [c.f0_std_st.mean() for c in cln], color=BLUE, zorder=3, s=28, label='only clips that stopped on their own')
    for i, c in enumerate(cln):
        ax.annotate(f'n={c.f0_std_st.notna().sum()}', (i, 0.2), ha='center', fontsize=7.5, color=MUT)
    ax.axhline(nat, color=ORANGE, ls='--', lw=1.2, label=f'natural recordings')
    ax.set_xticks(range(len(GEN)), [LBL[g] for g in GEN], fontsize=8); ax.set_title(name, fontsize=10, color=INK); grid(ax)
axes[0].set_ylabel('F0 standard deviation (semitones)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, ncol=3, loc='lower center', bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig('fig3_f0.pdf')

# Fig 4: CER (median), all clips
fig, ax = plt.subplots(figsize=(6.6, 2.9)); conds = ["natural"] + GEN; x = np.arange(len(conds))
for k, ((s, name), c) in enumerate(zip(SPK, (BLUE, ORANGE))):
    v = [d[(d.speaker == s) & (d.condition == g)].cer.median() for g in conds]
    bars = ax.bar(x + (k - .5) * (w + .02), v, w, color=c, label=name)
    for b, n in zip(bars, v):
        ax.annotate(f'{n:.2f}', (b.get_x() + b.get_width() / 2, n), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=8, color=INK)
ax.set_xticks(x, [LBL[g] for g in conds]); ax.set_ylabel('Median Whisper CER\n(lower = more intelligible)')
ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper left'); grid(ax)
fig.tight_layout(); fig.savefig('fig4_cer.pdf')


# Tables
def boot(a, b, seed=1234, repeats=2000):
    """Same algorithm as src/evaluation/report.paired_bootstrap; pairs with a missing value are dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); diff = (b - a)[ok]
    draws = np.random.default_rng(seed).integers(0, len(diff), size=(repeats, len(diff)))
    lo, hi = np.quantile(diff[draws].mean(axis=1), [.025, .975])
    return diff.mean(), lo, hi, len(diff)


METRICS = [("cer", "CER", 2), ("f0_std_st", "F0 SD (st)", 2), ("f0_range_st", "F0 range (st)", 1),
           ("duration_s", "Duration (s)", 1), ("f0_ccc", "F0 CCC", 2), ("ssim", "S-SIM", 2)]
PAIRS = [("base", "nat1"), ("base", "nat3"), ("base", "nat5"), ("syn3", "nat3"), ("nat1", "nat3"), ("nat5", "nat3")]
PL = {"nat1": "nat-1", "nat3": "nat-3", "nat5": "nat-5", "syn3": "syn-3", "base": "base"}
out = []
for s, name in SPK:
    out.append(r"\multicolumn{7}{@{}l}{\textbf{%s}}\\" % name)
    for a, b in PAIRS:
        cells = []
        for m, _, dp in METRICS:
            A = d[(d.speaker == s) & (d.condition == a)].set_index("utterance_id")[m]
            B = d[(d.speaker == s) & (d.condition == b)].set_index("utterance_id")[m].reindex(A.index)
            mu, lo, hi, n = boot(A.values, B.values)
            t = f"{mu:+.{dp}f} [{lo:+.{dp}f}, {hi:+.{dp}f}]"
            cells.append(r"\textbf{%s}" % t if (lo > 0 or hi < 0) else t)
        out.append(f"{PL[b]} $-$ {PL[a]} & " + " & ".join(cells) + r"\\")
    out.append(r"\midrule")
out = out[:-1]
open("tab_paired.tex", "w").write(r"\begin{tabular}{@{}lllllll@{}}" "\n" r"\toprule" "\n" r"\textbf{Comparison} & \textbf{CER} & \textbf{F0 SD (st)} & \textbf{F0 range (st)} & \textbf{Duration (s)} & \textbf{F0 CCC} & \textbf{S-SIM}\\" "\n" r"\midrule" "\n" + "\n".join(out) + "\n" r"\bottomrule\end{tabular}" "\n")

rows = []
for s, name in SPK:
    rows.append(r"\multicolumn{9}{@{}l}{\textbf{%s}}\\" % name)
    for g in ["natural"] + GEN:
        q = d[(d.speaker == s) & (d.condition == g)]
        cap = "--" if g == "natural" else f"{int(q.capped.sum())}/15"
        rows.append(" & ".join([LBL[g].replace("\n", " "), cap, f"{q.cer.median():.2f}", f"{q.f0_std_st.mean():.2f}",
                                f"{q.f0_range_st.mean():.1f}", f"{q.duration_s.mean():.1f}",
                                "--" if g == "natural" else f"{q.f0_ccc.mean():.2f}",
                                "--" if g == "natural" else f"{q.energy_ccc.mean():.2f}", f"{q.ssim.mean():.2f}"]) + r"\\")
    rows.append(r"\midrule")
open("tab_cells.tex", "w").write(r"\begin{tabular}{@{}lrrrrrrrr@{}}" "\n" r"\toprule" "\n" r"\textbf{Condition} & \textbf{Capped} & \textbf{CER} (med.) & \textbf{F0 SD} & \textbf{F0 range} & \textbf{Dur. (s)} & \textbf{F0 CCC} & \textbf{En. CCC} & \textbf{S-SIM}\\" "\n" r"\midrule" "\n" + "\n".join(rows[:-1]) + "\n" r"\bottomrule\end{tabular}" "\n")
print("figures + tables written")


# Fig 0: experimental setup diagram
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family'] = ['Helvetica']
NATC, SYNC, TXTC, TGT = '#2a78d6', '#1baf7a', '#e9e8e3', '#eb6834'
fig = plt.figure(figsize=(7.6, 3.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 45); ax.axis('off')


def box(x, y, w, h, fc, ec=None, txt='', fs=7.5, tc=INK, ls='-', lw=0.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.6', fc=fc, ec=ec or fc, lw=lw, ls=ls))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=tc, weight='bold' if bold else 'normal')


def arrow(x0, y0, x1, y1, c=MUT):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, color=c, lw=0.9))


# Condition rows
rows = [('base', 0, None, 'no shots: target text only (also the source of syn-3 shot audio)'),
        ('nat-1', 1, NATC, '1 natural shot: text + recording of line k+1'),
        ('nat-3', 3, NATC, '3 natural shots: lines k+1..k+3'),
        ('nat-5', 5, NATC, '5 natural shots: lines k+1..k+5'),
        ('syn-3', 3, SYNC, "same 3 texts as nat-3, audio = the model's own base clips of lines k+1..k+3")]
ys = [35.5, 28, 20.5, 13, 5.5]; px, pw, tw = 12, 4.2, 3.6
ax.text(2, 42, 'Prompt to Orpheus-hi-pt', fontsize=8.5, color=INK, weight='bold')
ax.text(72, 42, 'Output', fontsize=8.5, color=INK, weight='bold')
for (name, n, c, desc), y in zip(rows, ys):
    ax.text(2, y + 2, name, fontsize=9, color=INK, weight='bold', va='center')
    ax.text(px, y + 5.1, desc, fontsize=6.5, color=MUT, va='center')
    x = px
    for j in range(n):
        box(x, y, tw, 4, TXTC, txt='txt', fs=6.5); box(x + tw + .3, y, pw, 4, c, txt='wav', fs=6.5, tc='white')
        x += tw + pw + 1.2
    box(x, y, 6, 4, 'white', ec=TGT, txt='txt k', fs=7, tc=TGT, lw=1.2)
    arrow(x + 6.4, y + 2, 71.6, y + 2)
    box(72, y, 8.5, 4, 'white', ec=INK, txt='wav k', fs=7, lw=0.9, ls='--')

# Evaluation panel
box(84, 5.5, 14.5, 34, '#f6f5f1', ec=MUT, lw=.7)
for yy in [y + 2 for y in ys]:
    arrow(80.9, yy, 83.6, yy)
ax.text(91.25, 37, 'Compared with', fontsize=7.5, color=INK, weight='bold', ha='center', va='center')
ax.text(91.25, 34.6, 'natural recording', fontsize=7.5, color=INK, weight='bold', ha='center', va='center')
ax.text(91.25, 32.2, 'of line k', fontsize=7.5, color=INK, weight='bold', ha='center', va='center')
ax.text(91.25, 29.2, '(never in the prompt)', fontsize=6.5, color=MUT, ha='center', va='center')
ax.text(91.25, 18, 'Whisper CER\nF0 SD / range\nF0 / energy CCC\nspeaker similarity\n(S-SIM)', fontsize=7, color=INK,
        ha='center', va='center', linespacing=1.6)

# Legend
lx, ly = 2, 1.4
for c, t in [(TXTC, 'text of a line (LLM Devanagari)'), (NATC, 'natural recording'), (SYNC, "model's own audio (from base)")]:
    box(lx, ly - 1.1, 2.2, 2.2, c); ax.text(lx + 3, ly, t, fontsize=7, color=INK, va='center'); lx += 22
fig.savefig('fig0_setup.pdf')
