"""Descriptive completion tables; does not change the frozen M2 test family."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_m2 import ci


def aggregate(data,keys,count):
    metrics=['code_baseline_distance','code_normalized_recovery','code_normalized_selectivity','code_own_absolute_recovery',
        'code_selective_absolute_recovery','full_normalized_recovery','full_own_absolute_recovery','actual_delta_norm',
        'valid_phase_mass','total_stop_probability','delta_target_logp','competing_normalized_recovery']
    result=[]
    for key,g in data.groupby(keys):
        row=dict(zip(keys,key));expected=count(g);row.update(cells=len(g),expected_cells=expected,complete=len(g)==expected)
        for c in metrics:row[c]=float(g[c].mean()) if len(g)==expected and g[c].notna().all() else np.nan
        row['baseline_boundary_cells']=int(g.boundary_flag.sum());result.append(row)
    return pd.DataFrame(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);data=pd.read_csv(a.analysis/'per_intervention.csv')
    data=data[data.study_split=='confirmation'].copy()
    data['code_baseline_distance']=[r['code_baseline_to_'+r['donor']] for r in data.to_dict('records')]
    data['competing_normalized_recovery']=(data.code_own_absolute_recovery-data.code_selective_absolute_recovery)/data.code_baseline_distance
    data.loc[data.code_baseline_distance<.001,'competing_normalized_recovery']=np.nan
    common=['id','category','history','layer','site','comparison']
    by_donor=aggregate(data,common+['style','phase'],lambda g:1 if g.comparison.iloc[0] in ['shared','specific'] else 2)
    by_time=aggregate(data,common+['phase','frame'],lambda g:3)
    by_donor.to_csv(a.output/'per_utterance_donor_phase.csv',index=False);by_time.to_csv(a.output/'per_utterance_time.csv',index=False)
    rows=[]
    for key,g in by_donor.groupby(['history','layer','site','comparison','style','phase']):
        for metric in ['code_baseline_distance','code_normalized_recovery','code_normalized_selectivity','code_own_absolute_recovery','code_selective_absolute_recovery','competing_normalized_recovery']:
            rows.append(dict(zip(['history','layer','site','comparison','style','phase'],key),metric=metric,**ci(g[metric],g.category)))
    pd.DataFrame(rows).to_csv(a.output/'donor_phase_summaries.csv',index=False)
    selected=data[(data.history==0)&(data.layer==27)&(data.site=='mlp')&(data.phase==0)&(data.recipient=='news')]
    native=selected[selected.comparison=='native'];rows=[]
    pairs=pd.read_csv(a.analysis/'native_random_matched.csv');pairs=pairs[(pairs.study_split=='confirmation')&(pairs.history==0)&(pairs.layer==27)&(pairs.site=='mlp')&(pairs.phase==0)]
    for uid,g in native.groupby('id'):
        x=pairs[pairs.id==uid]
        rows.append(dict(id=uid,category=g.category.iloc[0],baseline_distance=g.code_baseline_distance.mean(),native_delta_norm=g.actual_delta_norm.mean(),
            selective_excess=x.code_normalized_selectivity_excess.mean() if len(x)==6 and x.code_normalized_selectivity_excess.notna().all() else np.nan,
            own_recovery=g.code_normalized_recovery.mean() if len(g)==6 and g.code_normalized_recovery.notna().all() else np.nan))
    texts=pd.DataFrame(rows);texts.to_csv(a.output/'distance_norm_effect_per_text.csv',index=False)
    correlations=[]
    for covariate in ['baseline_distance','native_delta_norm']:
        group=texts[[covariate,'selective_excess']].dropna();r=spearmanr(group[covariate],group.selective_excess)
        correlations.append(dict(covariate=covariate,n=len(group),spearman_rho=float(r.statistic),interpretation='Exploratory utterance-level association; no causal or multiplicity-adjusted claim'))
    pd.DataFrame(correlations).to_csv(a.output/'distance_norm_associations.csv',index=False)
    # Matched projection contrasts at the one frozen position, not differences of medians.
    pos=selected[selected.frame==4].pivot(index=['id','category','style'],columns='comparison',values='code_normalized_selectivity').reset_index()
    for method in ['specific','shared','native']:pos[method+'_minus_random']=pos[method]-pos['random']
    pos['native_minus_specific']=pos['native']-pos['specific'];pos['specific_minus_shared']=pos['specific']-pos['shared']
    pos.to_csv(a.output/'projection_matched_per_donor.csv',index=False)
    rows=[]
    for name in ['native_minus_random','specific_minus_random','shared_minus_random','native_minus_specific','specific_minus_shared']:
        values=[];categories=[]
        for uid,g in pos.groupby('id'):
            if len(g)==3 and g[name].notna().all():values.append(g[name].mean());categories.append(g.category.iloc[0])
        rows.append(dict(contrast=name,**ci(values,categories)))
    pd.DataFrame(rows).to_csv(a.output/'projection_matched_summary.csv',index=False)
    failures=[]
    for (style,phase),g in data[(data.history==0)&(data.layer==27)&(data.site=='mlp')&(data.comparison=='native')].groupby(['style','phase']):
        failures.append(dict(style=style,phase=phase,cells=len(g),negative_own_recovery_cells=int((g.code_own_absolute_recovery<0).sum()),
            negative_selectivity_cells=int((g.code_selective_absolute_recovery<0).sum()),baseline_boundary_cells=int(g.boundary_flag.sum()),
            patched_high_stop_cells=int((g.total_stop_probability>.01).sum()),patched_low_phase_mass_cells=int((g.valid_phase_mass<.99).sum())))
    pd.DataFrame(failures).to_csv(a.output/'selected_intervention_failure_diagnostics.csv',index=False)
    # Paired frame difference, with texts as units; descriptive, outside Holm family.
    all_pairs=pd.read_csv(a.analysis/'native_random_matched.csv')
    all_pairs=all_pairs[(all_pairs.study_split=='confirmation')&(all_pairs.history==0)&(all_pairs.phase==0)]
    frame_rows=[];frame_summary=[]
    for (layer,site),group in all_pairs.groupby(['layer','site']):
        rows=[]
        for uid,g in group.groupby('id'):
            if len(g)!=6 or not g.code_normalized_selectivity_excess.notna().all():continue
            by=g.groupby('frame').code_normalized_selectivity_excess.mean()
            item=dict(id=uid,category=g.category.iloc[0],layer=layer,site=site,later_minus_early=by[18]-by[4]);rows.append(item);frame_rows.append(item)
        frame=pd.DataFrame(rows)
        if len(frame):frame_summary.append(dict(layer=layer,site=site,**ci(frame.later_minus_early,frame.category)))
    pd.DataFrame(frame_rows).to_csv(a.output/'paired_frame_per_utterance.csv',index=False);pd.DataFrame(frame_summary).to_csv(a.output/'paired_frame_summary.csv',index=False)
    rows=[]
    for comp,g in selected[selected.frame==4].groupby('comparison'):
        for metric in ['code_normalized_recovery','code_normalized_selectivity','code_own_absolute_recovery','code_selective_absolute_recovery']:
            text=g.groupby(['id','category'],as_index=False)[metric].mean()
            rows.append(dict(comparison=comp,metric=metric,**ci(text[metric],text.category)))
    pd.DataFrame(rows).to_csv(a.output/'matched_frame4_own_and_selectivity.csv',index=False)
    effects=pd.read_csv(a.analysis/'per_utterance_phase_effects.csv')
    effects=effects[(effects.study_split=='confirmation')&(effects.history==0)&(effects.layer==27)&(effects.site=='mlp')&(effects.phase==0)]
    rows=[dict(category=name,**ci(g.code_normalized_selectivity_excess,g.category)) for name,g in effects.groupby('category')]
    pd.DataFrame(rows).to_csv(a.output/'text_category_summary.csv',index=False)
    (a.output/'ANALYSIS_SCOPE.md').write_text('These are descriptive/exploratory summaries of the frozen M2 cases. They add no GPU execution, hypothesis selection, new primary outcomes, or tests to the prespecified Holm family. Intervals resample utterances, holding the model, basis and seeds fixed.\n')


if __name__=='__main__':main()
