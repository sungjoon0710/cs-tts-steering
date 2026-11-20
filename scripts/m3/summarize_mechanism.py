"""Document-level development summaries and figures for fixed-history M3 replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MEASURES = [
    "code_own_absolute_recovery", "code_normalized_recovery",
    "code_selective_absolute_recovery", "code_normalized_selectivity",
    "full_own_absolute_recovery", "full_normalized_recovery",
    "valid_phase_mass", "total_stop_probability",
]


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0, np.nan, np.nan, np.nan
    rng = np.random.default_rng(20261118)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return len(values), float(values.mean()), *np.quantile(draws, [.025,.975])


def summarize(frame, keys, outcomes):
    results = []
    for key, group in frame.groupby(keys):
        key = key if isinstance(key,tuple) else (key,)
        for name in outcomes:
            n, mean, low, high = bootstrap(group.groupby("id")[name].mean())
            results.append(dict(zip(keys,key), outcome=name, n_documents=n,
                                mean=mean, ci_low=low, ci_high=high))
    return pd.DataFrame(results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--design", type=Path,
                   default=Path(__file__).resolve().parents[2] / "configs/research/m3/design.json")
    args=p.parse_args()
    data = pd.read_csv(args.input/"mechanism.csv")
    receipt = json.loads((args.input/"receipt.json").read_text())
    design_path = args.design
    design = json.loads(design_path.read_text())
    if hashlib.sha256(design_path.read_bytes()).hexdigest() != receipt["design_sha256"]:
        raise ValueError("Study design changed after running")
    for name, digest in receipt["output_hashes"].items():
        if hashlib.sha256((args.input/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Result hash changed")
    key = ["id","style","j","variant"]
    if data.duplicated(key).any() or set(data.id)!=set(design["assessment"]):
        raise ValueError("Incomplete or duplicate development results")
    if not (data.predicted_phase.eq(data.j%7) & data.consumed_phase.eq((data.j-1)%7)).all():
        raise ValueError("Phase indexing differs")
    args.output.mkdir()
    all_summary = summarize(data, ["variant"], MEASURES)
    per_donor = summarize(data, ["variant","style","j"], MEASURES)
    all_summary.to_csv(args.output/"mechanism_summary.csv", index=False)
    per_donor.to_csv(args.output/"mechanism_by_style_position.csv", index=False)
    family = []
    for kind in ("global","position"):
        fit = data[data.variant==kind+"_fit"].set_index(key[:-1]).sort_index()
        control = data[data.variant==kind+"_random"].set_index(key[:-1]).sort_index()
        if not fit.index.equals(control.index):
            raise ValueError("Unpaired controller control")
        paired = (fit[MEASURES]-control[MEASURES]).reset_index()
        paired["controller"] = kind
        paired["absolute_fit_own"] = fit.code_own_absolute_recovery.to_numpy()
        paired["positive_fit_own"] = paired.absolute_fit_own.gt(0)
        paired["norm_relative_error"] = (abs(fit.mlp_delta_norm-control.mlp_delta_norm) /
                                         fit.mlp_delta_norm.replace(0,np.nan)).to_numpy()
        if paired.norm_relative_error.max() > .01:
            raise ValueError("Realized random perturbation differs by more than 1%")
        family.append(paired)
    paired=pd.concat(family)
    paired.to_csv(args.output/"controller_control_pairs.csv", index=False)
    summarize(paired, ["controller"], ["code_own_absolute_recovery", "code_selective_absolute_recovery",
                                       "code_normalized_selectivity","positive_fit_own"]).to_csv(
                                           args.output/"controller_control_summary.csv",index=False)
    order = ["joint","native_mlp","bypass_only","upstream_residual_via_mlp","attention_via_mlp",
             "global_fit","global_random","position_fit","position_random"]
    names = ["Joint donor","Native MLP only","Donor bypass only","L26 residual via MLP",
             "L27 attention via MLP","Global mean","Global random","Position mean","Position random"]
    fig, axes = plt.subplots(1,2,figsize=(10.3,5),sharey=True,layout="constrained")
    for ax,outcome,title in zip(axes,["code_own_absolute_recovery","code_selective_absolute_recovery"],
                                ["Own-target recovery","Own minus competing-style recovery"]):
        values = all_summary[all_summary.outcome==outcome].set_index("variant").loc[order]*1
        for i,(_, row) in enumerate(values.iterrows()):
            ax.errorbar(row["mean"]*1000,i,xerr=[[1000*(row["mean"]-row.ci_low)],
                         [1000*(row.ci_high-row["mean"])]],fmt="o",capsize=3,
                        color="#155271" if i<5 else "#965323")
        ax.axvline(0,color="0.3",lw=1)
        ax.set_title(title)
        ax.set_xlabel("Phase-conditional JSD recovery (millinats)")
        ax.grid(axis="x",alpha=.2)
    axes[0].set_yticks(range(len(names)),names)
    axes[0].invert_yaxis()
    fig.suptitle("Indic-Speak late-block pathways on 12 development documents")
    fig.savefig(args.output/"late_pathways.pdf")
    fig.savefig(args.output/"late_pathways.png",dpi=215)
    plt.close(fig)


if __name__=="__main__":
    main()
