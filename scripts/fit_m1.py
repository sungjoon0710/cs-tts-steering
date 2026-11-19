"""Development-only M1 geometric fit and immutable confirmation gate. CPU only."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,read_jsonl,write_json
from src.mechanism_study import SITES,fit_structure,project_structure


def csvrows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def finite_mean(values):
    x=np.asarray(values,np.float64);x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def global_outcomes(subject,geometry):
    outcomes={'historical_news_anchor_alignment':finite_mean(geometry['historical_news']),
              'historical_default_anchor_alignment':finite_mean(geometry['historical_default'])}
    for k,site in enumerate(SITES):
        outcomes['shared_alignment_'+site]=finite_mean(geometry['shared_cosine'][:,k])
        outcomes['specificity_margin_'+site]=finite_mean(geometry['specific_margin'][:,k])
    layers=csvrows(subject/'h0/layer_phase_metrics.csv')
    wanted=lambda layer:[float(r['normalized_l2']) for r in layers if r['reference']=='news' and r['site']=='resid' and int(r['layer'])==layer]
    outcomes['residual_magnitude_L26_minus_L5']=finite_mean(wanted(26))-finite_mean(wanted(5))
    prob=csvrows(subject/'h0/probability_phase_metrics.csv')
    rows=[r for r in prob if r['reference']=='news']
    phase0=finite_mean([float(r['jsd_nats']) for r in rows if r['phase']=='0'])
    others=finite_mean([float(r['jsd_nats']) for r in rows if r['phase']!='0'])
    outcomes['phase0_minus_other_phases_JSD']=phase0-others
    outcomes['news_relative_JSD']=finite_mean([float(r['jsd_nats']) for r in rows])
    h=subject/'history_robustness.npz'
    hc=subject/'history_comparison.json'
    identical=json.loads(hc.read_text())['identical_inspected_prefix'] if hc.exists() else False
    outcomes['history_direction_cosine']=finite_mean(np.load(h,allow_pickle=False)['cosine']) if h.exists() and not identical else np.nan
    return outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;rows=read_jsonl(root/'inputs/cohort.jsonl');development=[r for r in rows if r['study_split']=='development']
    if any(not (root/'study'/r['id']/'subject_complete.json').exists() and not (root/'study'/r['id']/'failure.json').exists() for r in development):
        raise ValueError('Development includes unattempted subjects; confirmation remains locked')
    eligible=[r for r in development if (root/'study'/r['id']/'subject_complete.json').exists()]
    if len(eligible)<24 or any(sum(r['category']==cat for r in eligible)<8 for cat in {r['category'] for r in development}):
        raise ValueError('Insufficient complete development data')
    if any((root/'study'/r['id']/'subject_complete.json').exists() for r in rows if r['study_split']=='confirmation'):
        raise ValueError('Confirmation already executed; do not fit a new basis')
    a.output.mkdir(parents=True,exist_ok=False)
    deltas=[];hashes={}
    for row in eligible:
        path=root/'study'/row['id']/'h0/phase_contrasts.npz'
        done=json.loads(path.with_name('completion.json').read_text())
        if done['protocol_sha256']!=digest((root/'inputs/protocol.json').read_bytes()):raise ValueError('Development protocol mismatch')
        if digest(path.read_bytes())!=done['artifacts']['phase_contrasts.npz']:raise ValueError('Changed development tensor')
        with np.load(path,allow_pickle=False) as z:deltas.append(z['news_relative'].copy())
        hashes[row['id']]=digest(path.read_bytes())
    basis=fit_structure(np.stack(deltas));np.savez_compressed(a.output/'basis.npz',**basis)
    geometry=[];globals=[]
    for row,delta in zip(eligible,deltas):
        metrics=project_structure(delta,basis);geometry.append(metrics)
        primary=root/'study'/row['id']/'h0'
        with np.load(primary/'geometry_metrics.npz',allow_pickle=False) as z:other={k:v.copy() for k,v in z.items()}
        np.savez_compressed(a.output/(row['id']+'_geometry.npz'),**other,**metrics)
        globals.append({'id':row['id'],'category':row['category'],**global_outcomes(root/'study'/row['id'],{**other,**metrics})})
    shared=np.stack([r['shared_cosine'] for r in geometry]);specific=np.stack([r['specific_margin'] for r in geometry])
    # Mean across independent development texts, three labels and seven phases;
    # choose site/layer once, never after confirmation outcomes.
    shared_score=np.nanmean(shared,axis=(0,1,4));specific_score=np.nanmean(specific,axis=(0,1,4))
    choices={}
    for name,score in [('shared_structure',shared_score),('style_specific_structure',specific_score)]:
        site,layer=np.unravel_index(np.nanargmax(score),score.shape)
        choices[name]={'site':SITES[site],'layer':int(layer),'development_mean':float(score[site,layer]),
            'recommendation':'candidate for controlled M2 intervention, not demonstrated causal site' if score[site,layer]>0 else 'no positive directional evidence; do not claim a successful target'}
    phases=[]
    for phase in range(7):
        phases.append(finite_mean([float(r['jsd_nats']) for row in eligible for r in csvrows(root/'study'/row['id']/'h0/probability_phase_metrics.csv') if r['reference']=='news' and int(r['phase'])==phase]))
    choices['prediction_phase']={'phase':int(np.nanargmax(phases)),'development_mean_JSD':max(phases),'all_phase_means':phases}
    choices['limits']='Selected on development only. No intervention executed. Lexical style labels, not verified emotion semantics.'
    write_json(a.output/'M2_shortlist.json',choices)
    # NaNs retain explicit missing status in CSV; no zero imputation.
    with (a.output/'development_outcomes.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(globals[0]));w.writeheader();w.writerows(globals)
    gate={'development_ids':[r['id'] for r in eligible],'excluded_development_ids':[r['id'] for r in development if r not in eligible],
        'development_tensor_sha256':hashes,'basis_sha256':digest((a.output/'basis.npz').read_bytes()),
        'shortlist_sha256':digest((a.output/'M2_shortlist.json').read_bytes()),
        'cohort_sha256':digest((root/'inputs/cohort.jsonl').read_bytes()),'protocol_sha256':digest((root/'inputs/protocol.json').read_bytes()),
        'fitted_on_confirmation':False,'primary_histories_only':True,'rank_shared_consensus':1,
        'fit_code_sha256':digest(Path(__file__).read_bytes())}
    write_json(a.output/'confirmation_gate.json',gate)
    print(json.dumps({'development_n':len(eligible),'basis_sha256':gate['basis_sha256'],'confirmation_unlocked':True,'shortlist':choices}))


if __name__=='__main__':main()
