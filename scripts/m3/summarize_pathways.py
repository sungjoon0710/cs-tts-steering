"""Verify linked artifacts and create the lab-meeting tables and figure."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PLAN = Path(__file__).resolve().parents[2] / "configs/research/m3/plan.json"
OWN="code_own_absolute_recovery"
RATIO="code_normalized_recovery"
SELECT="code_selective_absolute_recovery"
OUTCOMES=[OWN,RATIO,SELECT,"code_normalized_selectivity","valid_phase_mass",
          "eos_128258_probability","eos_128001_probability","total_stop_probability"]
KEY=["id","style","j"]


def pointwise(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    batches=np.random.default_rng(191124).choice(values,(10000,len(values)),replace=True).mean(1)
    return dict(n=len(values),mean=values.mean(),low=np.quantile(batches,.025),high=np.quantile(batches,.975))


def read_verified(run):
    receipt=json.loads((run/"receipt.json").read_text())
    if receipt["status"]!="passed":
        raise ValueError("Exact engineering gate has not passed")
    for name,expected in receipt["output_hashes"].items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Artifact integrity failure: {name}")
    checks=pd.read_json(run/"engineering.jsonl",lines=True)
    if len(checks)!=receipt["engineering_checks"] or not checks.passed.all():
        raise ValueError("Failed or missing engineering checks")
    return receipt,checks


def pairs(result):
    pivot={name:result[result.condition==name].set_index(KEY).sort_index() for name in result.condition.unique()}
    if any(not t.index.is_unique or not pivot["joint"].index.equals(t.index) for t in pivot.values()):
        raise ValueError("Incomplete or non-unique doc/style/prediction pairing")
    effects={}
    for a,b in (("text_only","audio_only"),("joint","text_only"),("joint","audio_only"),
                ("boundaries_only","joint"),("joint_plus_boundaries","joint"),
                ("unrestricted","joint_plus_boundaries"),("unrestricted","boundaries_only")):
        effects[a+"_minus_"+b]=pivot[a][OUTCOMES]-pivot[b][OUTCOMES]
    for route in ("text_only","audio_only","joint","unrestricted"):
        effects[route+"_unclamped_minus_clamped"]=pivot[route][OUTCOMES]-pivot[route+"_clamped"][OUTCOMES]
        if route+"_random" in pivot:
            effects[route+"_minus_random"]=pivot[route][OUTCOMES]-pivot[route+"_random"][OUTCOMES]
    primary=[OWN,RATIO,SELECT,"code_normalized_selectivity"]
    effects["joint_factorial_recovery"]=(pivot["joint"][primary]-pivot["text_only"][primary]
                                                -pivot["audio_only"][primary]+pivot["nonstyle_donor"][primary])
    return effects


def make_figure(table,out):
    entries=[("text_only","Literal text"),("audio_only","Speech start + audio"),
             ("joint","Text + audio"),("boundaries_only","Delimiters only"),
             ("joint_plus_boundaries","Text + audio + delimiters"),
             ("unrestricted","Unrestricted early swap")]
    if "style_close_newline" in table.condition.to_numpy():
        entries[4:4]=[("style_close_newline","Style close + newline"),
                      ("end_text_controls","End-of-prompt controls")]
    order,labels=zip(*entries)
    colors={"patch":"#07557A","control":"#697580","clamp":"#9A461B"}
    fig,axes=plt.subplots(1,2,figsize=(10,2+.38*len(order)),sharey=True,layout="constrained")
    for ax,outcome,title,mult in zip(axes,(OWN,RATIO),("Absolute improvement (millinats)","Normalized recovery (%)"),(1000,100)):
        rows=table[table.outcome==outcome].set_index("condition")
        for i,base in enumerate(order):
            annotations=[(base,"patch","o",0)]
            if base+"_random" in rows.index:
                annotations.append((base+"_random","control","s",.19))
            if base+"_clamped" in rows.index:
                annotations.append((base+"_clamped","clamp","D",-.19))
            for name,group,fmt,shift in annotations:
                t=rows.loc[name]
                ax.errorbar(mult*t["mean"],i+shift,xerr=[[mult*(t["mean"]-t.low)],
                    [mult*(t.high-t["mean"])]],color=colors[group],fmt=fmt,capsize=3,ms=5)
        ax.axvline(0,color="0.35",linewidth=.8)
        ax.grid(axis="x",alpha=.19)
        ax.set_xlabel(title)
    axes[0].set_yticks(range(len(order)),labels)
    axes[0].invert_yaxis()
    for group,marker,lbl in (("patch","o","Donor K/V"),("clamp","D","L27 MLP clamped"),
                             ("control","s","Norm-matched random")):
        axes[1].plot([],[],marker=marker,color=colors[group],linestyle="none",label=lbl)
    axes[1].legend(loc="upper right",frameon=False,fontsize=8)
    fig.suptitle("Indic-Speak early style influence by attention-query position",
                 fontsize=13,weight="semibold")
    if "style_close_newline" in order:
        fig.supxlabel("Six development texts. Intervals resample documents. Delimiter split is exploratory.",
                      fontsize=8)
    fig.savefig(out/"pathway_comparison.png",dpi=220)
    fig.savefig(out/"pathway_comparison.pdf")
    plt.close(fig)


def main(run, out, plan_path=DEFAULT_PLAN):
    receipt,checks=read_verified(run)
    plan_bytes = plan_path.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != receipt["plan_sha256"]:
        raise ValueError("Study plan changed after running")
    plan = json.loads(plan_bytes)
    d=pd.read_csv(run/"pathways.csv")
    docs=list(d.id.unique())
    n_docs=len(docs)
    if (docs!=receipt["documents"] or len(d)!=n_docs*135
            or len(d.condition.unique())!=15 or set(d.j)!=set(plan["score_rows"])
            or (d.phase!=d.j%7).any()):
        raise ValueError("Incomplete causal contrasts")
    out.mkdir()
    summary=[]
    for name,g in d.groupby("condition"):
        for k in OUTCOMES:
            summary.append(dict(condition=name,outcome=k,**pointwise(g.groupby("id")[k].mean())))
    table=pd.DataFrame(summary)
    table.to_csv(out/"pathway_summary.csv",index=False)
    paired=[]; paired_summary=[]
    for name,frame in pairs(d).items():
        paired.append(frame.reset_index().assign(contrast=name))
        for column in frame:
            paired_summary.append(dict(contrast=name,outcome=column,
                **pointwise(frame[column].groupby("id").mean())))
    pd.concat(paired,ignore_index=True).to_csv(out/"paired_rows.csv",index=False)
    pd.DataFrame(paired_summary).to_csv(out/"paired_summary.csv",index=False)
    make_figure(table,out)
    tok=pd.read_csv(run/"prompt_tokens.csv",keep_default_na=False)
    obs=pd.read_csv(run/"prompt_observational.csv")
    if len(tok)!=n_docs*6 or len(obs) not in (n_docs*28,n_docs*32) or obs.duplicated(["id","target","reference","j"]).any():
        raise ValueError("Incomplete prompt audit")
    if any(len(ast.literal_eval(r.prompt_ids))!=r.prompt_tokens for r in tok.itertuples()):
        raise ValueError("Prompt lengths disagree with tokens")
    prompt_rows=[]
    for label,group in tok.groupby("label",sort=False):
        first=group.iloc[0]
        for key in ("style_token_ids","style_token_positions","tokens_vs_news"):
            if group[key].nunique()!=1:
                raise ValueError("Expected consistent prompt formatting")
        prompt_rows.append(dict(label=label,style=first["style"],token_ids=first.style_token_ids,
          zero_based_style_slots=first.style_token_positions, length_delta_vs_news=first.tokens_vs_news,
          total_prompt_tokens=f"{group.prompt_tokens.min()}–{group.prompt_tokens.max()}",
          d005_speech_start=first.speech_start))
    pd.DataFrame(prompt_rows).to_csv(out/"prompt_control_table.csv",index=False)
    summaries=[]
    for (target,ref),g in obs.groupby(["target","reference"]):
        for cohort,subset in (("interior",g[g.j>0]),("speech_start",g[g.j==0])):
            summaries.append(dict(target=target,reference=ref,rows=cohort,
                **pointwise(subset.groupby("id").conditional_jsd_nats.mean())))
    pd.DataFrame(summaries).to_csv(out/"prompt_observational_summary.csv",index=False)
    summary=dict(documents=n_docs,interventions=len(d),engineering_passes=len(checks),
         lowest_valid_mass=float(d.valid_phase_mass.min()),
         highest_total_stop_probability=float(d.total_stop_probability.max()),
         missing_normalized_recovery=int(d[RATIO].isna().sum()),
         max_random_norm_relative_error=float(d.random_norm_relative_error.dropna().max()))
    (out/"analysis_receipt.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    a=p.parse_args()
    main(a.run, a.output, a.plan)
