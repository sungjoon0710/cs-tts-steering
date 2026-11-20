"""Summarize the exploratory split without changing the frozen primary tables."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_pathways import KEY, OUTCOMES, OWN, RATIO, SELECT, make_figure, pointwise, read_verified


def main(main_run,followup,out,figure_directory=None):
    receipt,checks=read_verified(followup)
    primary=pd.read_csv(main_run/"pathways.csv").set_index([*KEY,"condition"]).sort_index()
    extension=pd.read_csv(followup/"pathways.csv").set_index([*KEY,"condition"]).sort_index()
    if not extension.index.is_unique or not primary.index.isin(extension.index).all():
        raise ValueError("Missing primary cases in the delimiter replication")
    if not np.array_equal(primary[OUTCOMES].to_numpy(),extension.loc[primary.index,OUTCOMES].to_numpy()):
        raise ValueError("Repeated primary endpoints differ in the exploratory replay")
    out.mkdir()
    order=("style_close_only","metadata_newline_only","style_close_newline",
           "end_text_controls","boundaries_only","style_close_newline_clamped",
           "end_text_controls_clamped","boundaries_only_clamped",
           "boundaries_only_random","boundaries_nonstyle_random")
    frames={name:extension.xs(name,level="condition") for name in order}
    rows=[]
    for name,frame in frames.items():
        for measure in OUTCOMES:
            rows.append(dict(condition=name,outcome=measure,**pointwise(frame[measure].groupby("id").mean())))
    delimiter_table=pd.DataFrame(rows)
    delimiter_table.to_csv(out/"delimiter_summary.csv",index=False)
    paired=[]
    for a,b in (("style_close_only","metadata_newline_only"),("style_close_newline","end_text_controls"),
                ("style_close_newline","boundaries_only"),
                ("boundaries_only","boundaries_only_clamped"),
                ("style_close_newline","style_close_newline_clamped"),
                ("end_text_controls","end_text_controls_clamped"),
                ("boundaries_only","boundaries_only_random"),
                ("boundaries_only","boundaries_nonstyle_random")):
        for metric in (OWN,RATIO,SELECT):
            effect=frames[a][metric]-frames[b][metric]
            paired.append(dict(contrast=a+"_minus_"+b,outcome=metric,**pointwise(effect.groupby("id").mean())))
    pd.DataFrame(paired).to_csv(out/"delimiter_paired_summary.csv",index=False)
    mapping={
        "joint":("text_only","audio_only"),
        "joint_plus_boundaries":("joint","boundaries_only"),
        "boundaries_only":("style_close_newline","end_text_controls")}
    additivity=[]
    for name,(a,b) in mapping.items():
        both=extension.xs(name,level="condition")
        one=extension.xs(a,level="condition")
        other=extension.xs(b,level="condition")
        for m in (OWN,RATIO,SELECT):
            residual=both[m]-one[m]-other[m]
            additivity.append(dict(condition=name,measure="factorial_"+m,**pointwise(residual.groupby("id").mean())))
        for m in ("joint_vs_log_additive_jsd_nats","joint_vs_news_conditional_jsd_nats"):
            additivity.append(dict(condition=name,measure=m,**pointwise(both[m].groupby("id").mean())))
    pd.DataFrame(additivity).to_csv(out/"log_additivity.csv",index=False)
    (out/"verification.txt").write_text(f"{len(checks)} exact checks passed on {len(receipt['documents'])} documents\n"
         f"{len(primary)} primary rows repeated with identical reported endpoints\n"
         f"{len(extension)} follow-up rows\n")
    if figure_directory:
        update=delimiter_table[delimiter_table.condition!="boundaries_only"]
        core=pd.read_csv(figure_directory/"pathway_summary.csv")
        make_figure(pd.concat([core,update],ignore_index=True),figure_directory)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("main_run","followup","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--figure-directory",type=Path)
    a=p.parse_args()
    main(a.main_run,a.followup,a.output,a.figure_directory)
