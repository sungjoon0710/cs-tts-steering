"""Utterance-level M2 analysis and development-only confirmation-design freeze."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest
from src.mechanism_study import holm
from scripts.analyze_m1 import bootstrap_mean

BRANCHES=[(5,'attn'),(11,'attn'),(11,'mlp'),(27,'mlp')]


def load(root):
    records=[];coverage=[];lineage=[]
    for done in sorted((root/'runs').glob('*/*/h*/completion.json')):
        metadata=json.loads(done.read_text());folder=done.parent
        p=folder/'results.json'
        if digest(p.read_bytes())!=metadata['artifacts']['results.json']:raise ValueError('M2 results hash changed')
        if (folder/'probabilities.npz').exists() and digest((folder/'probabilities.npz').read_bytes())!=metadata['artifacts']['probabilities.npz']:
            raise ValueError('M2 probability archive changed')
        rows=json.loads(p.read_text())
        if len(rows)!=metadata['cases']:raise ValueError('Intervention case count differs')
        for row in rows:
            if row['earlier_prediction_max_abs_diff']!=0 or not row['hooks_removed'] or row['hook_calls']!=1:raise ValueError('Engineering invariance failed')
        records.extend(rows);coverage.append(dict(id=metadata['id'],history=metadata['history'],plan=folder.parents[1].name,cases=len(rows),runtime_s=metadata['runtime_s']))
        lineage.append(dict(path=str(done),sha256=digest(done.read_bytes()),**metadata))
    if not records:raise ValueError('No completed M2 causal results')
    data=pd.DataFrame(records)
    if data.duplicated(['id','history','case_id']).any():raise ValueError('Duplicate M2 physical case; do not count twice')
    return data,coverage,lineage


def ratios(data,floor):
    data=data.copy()
    for metric in ['full','code']:
        base=np.array([r[f'{metric}_baseline_to_{r["donor"]}'] for r in data.to_dict('records')])
        keep=base>=floor
        data[metric+'_normalized_recovery']=np.where(keep,data[metric+'_own_absolute_recovery']/base,np.nan)
        data[metric+'_normalized_selectivity']=np.where(keep,data[metric+'_selective_absolute_recovery']/base,np.nan)
    return data


def matched(data):
    keys=['id','study_split','category','history','layer','site','j','phase','frame','style']
    forward=data[data.recipient=='news']
    columns=keys+['code_normalized_recovery','code_normalized_selectivity','full_normalized_recovery','full_normalized_selectivity',
                  'code_own_absolute_recovery','code_selective_absolute_recovery','actual_delta_norm','boundary_flag']
    native=forward[forward.comparison=='native'][columns]
    random=forward[forward.comparison=='random'][columns]
    pairs=native.merge(random,on=keys,suffixes=('_native','_random'),validate='one_to_one')
    for metric in ['code_normalized_recovery','code_normalized_selectivity','full_normalized_recovery','full_normalized_selectivity',
                   'code_own_absolute_recovery','code_selective_absolute_recovery']:
        pairs[metric+'_excess']=pairs[metric+'_native']-pairs[metric+'_random']
    norm=pairs.actual_delta_norm_native.to_numpy()
    error=np.abs(pairs.actual_delta_norm_random-norm)/np.where(norm>0,norm,1)
    if np.any(error>.0101):raise ValueError('Random and native realized norms are not matched')
    return pairs


def per_text(pairs):
    keys=['id','study_split','category','history','layer','site','phase']
    metrics=[c for c in pairs if c.endswith('_excess')]
    result=[]
    for key,g in pairs.groupby(keys,sort=True):
        item=dict(zip(keys,key));item['n_cells']=len(g)
        complete=len(g)==6 and set(g['style'])=={'anger','happy','sad'} and set(g['frame'])=={4,18} and not g.duplicated(['style','frame']).any()
        for metric in metrics:
            # All three donors and both fixed positions, never varying denominators silently.
            item[metric]=float(g[metric].mean()) if complete and g[metric].notna().all() else np.nan
        item['boundary_cells']=int((g.boundary_flag_native|g.boundary_flag_random).sum())
        result.append(item)
    return pd.DataFrame(result)


def ci(values,categories,seed=20261116):
    x=np.asarray(values,float);valid=np.isfinite(x);x=x[valid];categories=np.asarray(categories)[valid].tolist()
    if not len(x):return dict(n=0,mean=np.nan,median=np.nan,ci_lower=np.nan,ci_upper=np.nan)
    b=bootstrap_mean(x,categories,2000,seed)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)) if len(x)>1 else np.nan,
                ci_lower=float(b['lower'][0]),ci_upper=float(b['upper'][0]))


def development(root,data,pairs,out,args):
    primary=data[(data.study_split=='development')&(data.history==0)]
    candidate=primary[(primary.recipient=='news')&(primary.phase==0)].drop_duplicates(['id','history','j','style'])
    distances=np.array([r[f'code_baseline_to_{r["donor"]}'] for r in candidate.to_dict('records')]);distances=distances[distances>0]
    floor=max(1e-6,min(.001,float(np.quantile(distances,.05))/10))
    data=ratios(data,floor);pairs=matched(data);pt=per_text(pairs);summaries=[]
    for layer,site in BRANCHES:
        values=pt[(pt.study_split=='development')&(pt.history==0)&(pt.layer==layer)&(pt.site==site)&(pt.phase==0)]
        if not len(values):continue
        r=dict(layer=layer,site=site,**ci(values.code_normalized_selectivity_excess,values.category))
        r['mean_own_recovery_excess']=float(values.code_normalized_recovery_excess.mean())
        cells=pairs[(pairs.study_split=='development')&(pairs.history==0)&(pairs.layer==layer)&(pairs.site==site)&(pairs.phase==0)]
        for metric in ['code_normalized_recovery_native','code_normalized_selectivity_native']:
            valid=[g[metric].mean() for _,g in cells.groupby('id') if len(g)==6 and g[metric].notna().all()]
            r['mean_'+metric]=float(np.mean(valid)) if valid else np.nan
        summaries.append(r)
    pd.DataFrame(summaries).to_csv(out/'development_branch_selection.csv',index=False)
    complete=len({r['id'] for r in primary.to_dict('records')})
    if args.freeze:
        # Only completed A and B across all 30 requested development subjects justify selection.
        a={p.parent.parent.name for p in (root/'runs/development_A').glob('*/h0/completion.json')}
        b={p.parent.parent.name for p in (root/'runs/development_B').glob('*/h0/completion.json')}
        if len(a)!=30 or a!=b:raise ValueError('Freeze requires all original 30 A/B development subjects, with no failed reference replacement')
        eligible=[r for r in summaries if r['n']>=24 and r['mean']>=.10 and r['ci_lower']>0 and r['mean_own_recovery_excess']>0
                  and r['mean_code_normalized_recovery_native']>=.10 and r['mean_code_normalized_selectivity_native']>0]
        best=max(eligible,key=lambda r:(r['mean'],-r['layer'],r['site'])) if eligible else None
        # 90% upper variability bound and conservative ten-test planning alpha.
        sd=max([r.get('sd',0) for r in summaries if np.isfinite(r.get('sd',np.nan))]+[.05])
        upper=sd*np.sqrt(29/stats.chi2.ppf(.1,29));required=int(np.ceil(((stats.norm.ppf(1-.05/20)+stats.norm.ppf(.8))*upper/.10)**2))
        n=min(80,max(60,required));decision=dict(normalization_floor=floor,minimum_effect=.10,development_n=30,development_sd_upper=upper,
            required_n_normal_approximation=required,confirmation_n=n,power_target=.8,planning_family_max=10,
            cap_underpowered_for_minimum_effect=required>80,selected_branch={k:best[k] for k in ['layer','site']} if best else None,
            selection_rule='>=24 texts; mean selective excess>=.10, CI lower>0, own excess>0, native own recovery>=.10 and native selectivity>0; select largest mean',
            phase_extension=bool(best),source_results_hashes={str(p.relative_to(root)):digest(p.read_bytes()) for p in (root/'runs').glob('development_*/*/h0/results.json')},
            fitted_on_confirmation=False)
        target=args.freeze
        if target.exists():raise FileExistsError('Never overwrite a development decision')
        target.write_text(json.dumps(decision,indent=2)+'\n')
    return data,pairs,pt,summaries


def confirmation_outcomes(data,pt,gate):
    outcome={};categories={}
    conf=data[(data.study_split=='confirmation')&(data.history==0)]
    for uid,g in conf.groupby('id'):outcome[uid]={};categories[uid]=g.category.iloc[0]
    def assign(name,layer,site,history=0):
        frame=pt[(pt.study_split=='confirmation')&(pt.history==history)&(pt.layer==layer)&(pt.site==site)&(pt.phase==0)]
        for _,r in frame.iterrows():outcome.setdefault(r.id,{})[name]=r.code_normalized_selectivity_excess;categories[r.id]=r.category
    for l,s in BRANCHES:assign(f'L{l}_{s}_selective_excess',l,s)
    assign('second_history_L5_attn_selective_excess',5,'attn',1)
    best=gate.get('selected_branch')
    if best:
        if best!={'layer':5,'site':'attn'}:assign('second_history_selected_selective_excess',best['layer'],best['site'],1)
        p=pt[(pt.study_split=='confirmation')&(pt.history==0)&(pt.layer==best['layer'])&(pt.site==best['site'])]
        for uid,g in p.groupby('id'):
            v=g.set_index('phase').code_normalized_selectivity_excess
            if all(k in v and np.isfinite(v[k]) for k in [0,1,2,4,5]):
                outcome[uid]['coarse_minus_fine_selective_excess']=v[0]-(v[2]+v[5])/2
                outcome[uid]['middle_minus_fine_selective_excess']=(v[1]+v[4]-v[2]-v[5])/2
    native=conf[(conf.site=='resid')&(conf.comparison=='native')&(conf.phase==0)]
    for uid,g in native.groupby('id'):
        means={l:float(v.code_normalized_recovery.mean()) if len(v)==6 and v.code_normalized_recovery.notna().all() else np.nan for l,v in g.groupby('layer')}
        outcome[uid]['residual_L26_minus_L5_recovery']=means.get(26,np.nan)-means.get(5,np.nan)
        outcome[uid]['residual_L11_minus_L5_recovery']=means.get(11,np.nan)-means.get(5,np.nan)
    return [dict(id=uid,category=categories[uid],**r) for uid,r in outcome.items()]


def analyze(args):
    root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=False);data,coverage,lineage=load(root)
    gate=json.loads(args.gate.read_text()) if args.gate else None
    data=ratios(data,gate['normalization_floor'] if gate else 1e-6);pairs=matched(data);pt=per_text(pairs)
    summaries=[]
    if not gate:data,pairs,pt,summaries=development(root,data,pairs,out,args)
    data.to_csv(out/'per_intervention.csv',index=False);pairs.to_csv(out/'native_random_matched.csv',index=False);pt.to_csv(out/'per_utterance_phase_effects.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'execution_coverage.csv',index=False)
    all_summary=[]
    for key,g in data.groupby(['study_split','history','layer','site','comparison','phase']):
        # Mean fixed donors/positions within text first, not pseudo-replicated patches.
        for metric in ['code_normalized_recovery','code_normalized_selectivity','code_own_absolute_recovery','code_selective_absolute_recovery']:
            grouped=g.groupby(['id','category'],as_index=False)[metric].mean()
            all_summary.append(dict(zip(['study_split','history','layer','site','comparison','phase'],key),metric=metric,
                                    **ci(grouped[metric],grouped.category)))
    pd.DataFrame(all_summary).to_csv(out/'site_control_phase_summary.csv',index=False)
    if gate:
        records=confirmation_outcomes(data,pt,gate);frame=pd.DataFrame(records);frame.to_csv(out/'confirmation_per_utterance.csv',index=False)
        tests=[];pvalues=[]
        for name in gate['primary_outcomes']:
            v=frame[name].to_numpy(float) if name in frame else np.full(len(frame),np.nan);keep=np.isfinite(v);x=v[keep]
            if len(x)<2:raw=1.;stat=np.nan;status='insufficient complete utterances'
            elif np.std(x,ddof=1)==0:raw=1.;stat=np.nan;status='degenerate sample; no variance-based test'
            else:t=stats.ttest_1samp(x,0);raw=float(t.pvalue);stat=float(t.statistic);status='tested'
            pvalues.append(raw);tests.append(dict(outcome=name,status=status,t=stat,p_raw=raw,requested_n=20 if name.startswith('second_history') else gate['confirmation_n'],
                **ci(v,frame.category)))
        for r,p in zip(tests,holm(pvalues)):r['p_holm']=float(p);r['reject_at_05']=bool(p<.05 and r['status']=='tested')
        pd.DataFrame(tests).to_csv(out/'primary_confirmation_tests.csv',index=False)
    manifest=dict(results_files=len(lineage),cases=len(data),unique_texts=data.id.nunique(),normalization_floor=gate['normalization_floor'] if gate else None,
        gate_sha256=digest(args.gate.read_bytes()) if gate else None,source_sha256=digest(Path(__file__).read_bytes()),confirmation_fit_performed=False,
        probability_archives_available_locally=sum((Path(r['path']).parent/'probabilities.npz').exists() for r in lineage),
        lineage=lineage)
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='lineage'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path);p.add_argument('--freeze',type=Path);analyze(p.parse_args())
