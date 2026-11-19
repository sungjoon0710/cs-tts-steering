"""CPU-only integrity and independent probability checks for an owned M2 run."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,read_jsonl,write_json
from scripts.run_m2 import case_list
from scripts.run_m1 import valid_codes


def jsd(p,q):
    m=(p+q)/2
    return .5*(np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)/np.maximum(m,1e-300)),0))+
               np.sum(np.where(q>0,q*np.log(np.maximum(q,1e-300)/np.maximum(m,1e-300)),0)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--basis',type=Path,required=True)
    a=p.parse_args();root=a.root;inventory=[];coverage=[];max_error=0.;max_probability_sum_error=0.;independent=0;excluded=[];cases=0
    gate=json.loads((root/'inputs/confirmation_gate_verified.json').read_text())
    if digest(a.basis.read_bytes())!=gate['basis_sha256']:raise ValueError('Original frozen M1 basis changed')
    frozen={'cohort_primary':'confirmation_cohort.jsonl','cohort_secondary':'secondary_cohort.jsonl','protocol':'protocol_confirmation.json',
            'plan_primary':'plan_confirmation.json','plan_secondary':'plan_secondary.json'}
    for key,name in frozen.items():
        if digest((root/'inputs'/name).read_bytes())!=gate[key+'_sha256']:raise ValueError('Frozen input hash mismatch: '+name)
    checked_sources=0
    for manifest in (root/'code').glob('*/source_manifest.json'):
        hashes=json.loads(manifest.read_text())['sha256']
        if hashes.get('scripts/run_m2.py')!=gate['run_script_sha256']:continue
        if hashes.get('src/causal_patch.py')!=gate['patch_engine_sha256']:raise ValueError('Frozen engine hash differs')
        for name,sha in hashes.items():
            if digest((manifest.parent/name).read_bytes())!=sha:raise ValueError('Immutable source release changed')
        checked_sources+=1
    if not checked_sources:raise ValueError('No verified frozen runner/engine release')
    cohorts={r['id']:r for name in ['development_cohort.jsonl','confirmation_cohort.jsonl'] for r in read_jsonl(root/'inputs'/name)}
    stages={name:json.loads((root/'inputs'/file).read_text()) for name,file in [('development_A','plan_A.json'),('development_B','plan_B.json'),
        ('development_C','plan_C.json'),('confirmation','plan_confirmation.json')]}
    controls=json.loads((root/'engineering/engineering.json').read_text())
    if controls['status']!='passed' or any(controls[k]!=0 for k in ['no_hook_max_abs_diff','identity_max_abs_diff','zero_max_abs_diff']):raise ValueError('Original engineering gate failed')
    history_checks=0
    for meta in sorted((root/'baselines').glob('*/h*/metadata.json')):
        r=json.loads(meta.read_text());array=meta.parent/'baseline.npz'
        if digest(array.read_bytes())!=r['array_sha256']:raise ValueError('Baseline changed')
        if len({len(v) for v in r['prompts'].values()})!=1:raise ValueError('Unequal styled prompt lengths')
        codes=r['inspected_codes']
        if len(codes)!=168 or any(not 128266+j%7*4096<=v<128266+(j%7+1)*4096 for j,v in enumerate(codes)):raise ValueError('Code grammar mismatch')
        inventory.append(dict(path=str(array.relative_to(root)),bytes=array.stat().st_size,sha256=r['array_sha256'],type='baseline_logits_and_exact_states'))
        row=cohorts[r['id']]
        if r['text_sha256']!=row['text_sha256'] or r['checkpoint']!='76e0814f189321efe550011850cd288dae366aad' or r['codec']!='d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec':raise ValueError('Baseline identity differs')
        history=r['history_provenance']
        if valid_codes(history)[:168]!=codes:raise ValueError('Baseline history differs from original valid code prefix')
        if r['study_split']=='development':
            original=Path(history['m1_source'])
            if digest(original.read_bytes())!=history['m1_source_sha256']:raise ValueError('Original M1 history changed')
        else:
            original=root/'histories'/r['id']/f'history_{r["history"]}.json'
            if json.loads(original.read_text())!=history:raise ValueError('Original M2 history differs from cached provenance')
            if history['seed']!=[1234,5678][r['history']]:raise ValueError('Reference seed differs')
        history_checks+=1
    for done in sorted((root/'runs').glob('*/*/h*/completion.json')):
        meta=json.loads(done.read_text());folder=done.parent;records=json.loads((folder/'results.json').read_text());cases+=len(records)
        for name,sha in meta['artifacts'].items():
            path=folder/name
            if digest(path.read_bytes())!=sha:raise ValueError('Causal artifact changed: '+str(path))
            inventory.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha,type='intervention_artifact'))
        omissions=json.loads((folder/'exclusions.json').read_text()) if (folder/'exclusions.json').exists() else []
        excluded+=omissions
        if len(records)!=meta['cases'] or len(records)+len(omissions)!=meta.get('planned_cases',meta['cases']):raise ValueError('Case accounting mismatch')
        stage=folder.parents[1].name
        plan=json.loads((root/'inputs/plan_secondary.json').read_text()) if stage=='confirmation' and meta['history']==1 else stages[stage]
        expected={r['case_id']:r for r in case_list(plan)}
        if set(expected)!={r['case_id'] for r in records+omissions} or len(records+omissions)!=len(expected):raise ValueError('Frozen intervention coverage differs')
        for record in records+omissions:
            if any(record[k]!=v for k,v in expected[record['case_id']].items()):raise ValueError('Incorrect frozen case semantics')
        name='plan_secondary.json' if stage=='confirmation' and meta['history']==1 else 'plan_confirmation.json' if stage=='confirmation' else 'plan_'+stage[-1]+'.json'
        if meta['plan_sha256']!=digest((root/'inputs'/name).read_bytes()):raise ValueError('Plan provenance differs')
        if stage=='confirmation' and (meta['source_sha256']!=gate['run_script_sha256'] or meta['protocol_sha256']!=gate['protocol_sha256']):raise ValueError('Frozen run source/protocol differs')
        with np.load(folder/'probabilities.npz',allow_pickle=False) as z:probs=z['probabilities']
        if probs.shape!=(len(records),156960) or not np.isfinite(probs).all() or probs.min()<0:raise ValueError('Invalid full probabilities')
        err=float(np.max(np.abs(probs.astype(np.float64).sum(1)-1)));max_probability_sum_error=max(err,max_probability_sum_error)
        if err>2e-6:raise ValueError('Probability normalization failed')
        baseline=root/'baselines'/meta['id']/('h'+str(meta['history']))
        bmeta=json.loads((baseline/'metadata.json').read_text());rows=bmeta['prediction_rows']
        if meta['baseline_context_key']!=bmeta['context_key']:raise ValueError('Mismatched baseline history')
        # Independently recompute representative native/random/reverse and projected rows,
        # using probabilities rather than the original Torch log-probability implementation.
        chosen=[]
        for kind in ['native','random','reverse','shared','specific']:
            found=next((i for i,r in enumerate(records) if r['comparison']==kind),None)
            if found is not None:chosen.append(found)
        with np.load(baseline/'baseline.npz',allow_pickle=False) as z:
            for i in chosen:
                r=records[i];ix=rows.index(r['j']);lp=z[r['donor']+'__logits'][ix].astype(float);p0=np.exp(lp-lp.max());p0/=p0.sum();patched=probs[i].astype(float)
                e=abs(jsd(patched,p0)-r['full_patched_to_'+r['donor']]);offset=128266+r['phase']*4096
                q=patched[offset:offset+4096];q=q/q.sum();q0=p0[offset:offset+4096];q0=q0/q0.sum()
                e=max(e,abs(jsd(q,q0)-r['code_patched_to_'+r['donor']]));max_error=max(max_error,e);independent+=1
                if e>2e-6:raise ValueError('Independent JSD recomputation failed')
        for r in records:
            if r['earlier_prediction_max_abs_diff']!=0 or r['hook_calls']!=1 or not r['hooks_removed'] or not r['fresh_uncached']:raise ValueError('Engineering control failed')
            if r['j']!=7*r['frame']+r['phase'] or r['predicted_phase']!=r['phase'] or r['processed_phase']!=(r['j']-1)%7:raise ValueError('Position/phase mismatch')
            if r['absolute_position']!=len(bmeta['prompts'][r['recipient']])-1+r['j']:raise ValueError('Prompt offset mismatch')
            if not r['expected_recipient_exact'] or r['dtype']!='torch.bfloat16':raise ValueError('Recipient shape/dtype identity failed')
            if r['last_residual_donor_logit_max_abs_diff'] is not None and r['last_residual_donor_logit_max_abs_diff']>1e-5:raise ValueError('Final residual positive control failed')
        coverage.append(dict(id=meta['id'],history=meta['history'],plan=folder.parents[1].name,cases=len(records),excluded=len(omissions)))
    requested=[]
    for cohort,h in [('confirmation_cohort.jsonl',0),('secondary_cohort.jsonl',1)]:
        for row in read_jsonl(root/'inputs'/cohort):
            found=[r for r in coverage if r['id']==row['id'] and r['history']==h]
            fail=root/'failures'/f'reference_{row["id"]}_h{h}.json'
            state='complete' if found else 'unusable_reference' if fail.exists() else 'not_completed'
            requested.append(dict(id=row['id'],history=h,status=state,source_document=row['source_document'],category=row['category'],failure=json.loads(fail.read_text()) if fail.exists() else None))
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'verification.json',dict(status='passed',cases=cases,completed_histories=len({(r['id'],r['history']) for r in coverage}),
        requested_confirmation_histories=len(requested),confirmation_complete=sum(r['status']=='complete' for r in requested),
        confirmation_unusable_reference=sum(r['status']=='unusable_reference' for r in requested),confirmation_not_completed=sum(r['status']=='not_completed' for r in requested),
        independently_recomputed_rows=independent,max_independent_JSD_absolute_error=max_error,max_probability_sum_error=max_probability_sum_error,
        projected_norm_exclusions=len(excluded),probability_and_baseline_bytes=sum(r['bytes'] for r in inventory),source_sha256=digest(Path(__file__).read_bytes()),
        original_history_identity_checks=history_checks,frozen_gate_sha256=digest((root/'inputs/confirmation_gate_verified.json').read_bytes()),
        original_engineering_controls=controls['status'],frozen_case_semantics_verified=True,verified_source_releases=checked_sources))
    write_json(a.output/'requested_coverage.json',requested);write_json(a.output/'artifact_inventory.json',inventory);write_json(a.output/'projected_norm_exclusions.json',excluded)
    print((a.output/'verification.json').read_text())


if __name__=='__main__':main()
