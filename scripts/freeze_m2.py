"""Freeze independent M2 confirmation using completed development evidence only."""
import argparse,json,random,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,read_jsonl,write_json,write_jsonl
from scripts.prepare_m1 import category,conllu
from scripts.analyze_m2 import load,ratios,matched,per_text,confirmation_outcomes
from scripts.run_m2 import case_list


def canonical(text):return ' '.join(re.findall('[a-z]+',text.lower()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','m1','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();out=a.output
    if (out/'confirmation_gate.json').exists():raise FileExistsError('Confirmation design is immutable')
    decision=json.loads((out/'development_decision_AB.json').read_text())
    data,coverage,lineage=load(a.root)
    for stage in ['A','B','C']:
        if len([r for r in coverage if r['plan']=='development_'+stage])!=30:raise ValueError('All 30 development A/B/C histories must finish')
    data=ratios(data,decision['normalization_floor']);pt=per_text(matched(data))
    # Use the same fixed outcomes for development variability, without changing selection.
    dev=data.copy();dev['study_split']='confirmation';dpt=pt.copy();dpt['study_split']='confirmation'
    outcomes=pd.DataFrame(confirmation_outcomes(dev,dpt,decision))
    names=[c for c in outcomes if c not in ('id','category') and not c.startswith('second_history')]
    sd=max(float(outcomes[c].std()) for c in names)
    upper=sd*np.sqrt(29/stats.chi2.ppf(.1,29))
    required=int(np.ceil(((stats.norm.ppf(1-.05/20)+stats.norm.ppf(.8))*upper/.10)**2))
    n=min(80,max(60,decision['confirmation_n'],required));n+=n%2
    prior_path=a.m1/'remote_export_final_v1/inputs/prior_cohort.jsonl';m1_path=a.m1/'dataset/cohort.jsonl'
    prior=read_jsonl(prior_path)+read_jsonl(m1_path)
    docs={r['source_document'] for r in prior};texts={canonical(r.get('text',r.get('reply',''))) for r in prior}
    source=a.m1/'data/en_ewt-ud-dev.conllu';pool=[]
    for r in conllu(source):
        text=r['text'];words=re.findall('[A-Za-z]+',text)
        if not text.isascii() or not 8<=len(words)<=30 or len(text)>220 or re.search(r'\d|https?://|www\.|@',text) or not text.endswith(('.','!','?')):continue
        if r['source_document'] in docs or canonical(text) in texts:continue
        cat,evidence=category(text)
        pool.append(dict(r,category=cat,category_evidence=evidence,text_sha256=digest(text.encode()),canonical_text_sha256=digest(canonical(text).encode()),
            source='UD_English-EWT_r2.18',source_revision='b7711cce01cdd4f5fcc0a8199b8a50d951b16c0c',
            genre=r['source_document'].split('-')[0],study_split='confirmation',prior_exposure='No matching prior Nov12/M1 document or canonical text'))
    rng=random.Random(2026111602);selected=[]
    counts={c:len({r['source_document'] for r in pool if r['category']==c}) for c in ['emotion_marked','referential_context','lexical_control']}
    # Scarce emotion-marked documents were exhausted by M1. Do not manufacture them.
    for cat in ['referential_context','lexical_control']:
        group=sorted([r for r in pool if r['category']==cat],key=lambda r:(r['source_document'],r['source_id']));rng.shuffle(group);added=0
        for r in group:
            if r['source_document'] in docs or canonical(r['text']) in texts:continue
            docs.add(r['source_document']);texts.add(canonical(r['text']));selected.append(r);added+=1
            if added==n//2:break
        if added!=n//2:raise ValueError('Insufficient independent documents; do not reuse exposed samples')
    for i,r in enumerate(selected,1):r['id']=f'm{i:03d}';r['second_history']=False
    secondary=[]
    for cat in ['referential_context','lexical_control']:
        group=[r for r in selected if r['category']==cat];rng.shuffle(group)
        for r in group[:10]:r['second_history']=True;secondary.append(r)
    secondary.sort(key=lambda r:r['id'])
    assert len({r['source_document'] for r in selected})==n
    write_jsonl(out/'confirmation_cohort.jsonl',selected);write_jsonl(out/'secondary_cohort.jsonl',secondary)
    protocol=json.loads((out/'protocol_development.json').read_text());protocol.update(version=2,normalization_floor=decision['normalization_floor'],
        design_role='Frozen independent causal confirmation',confirmation_n=n,secondary_history_n=20,
        categories='Operational referential/context-linked and lexical controls; no remaining emotion-marked documents. Not gold emotional states.',
        statistic='Equal-weight three donors and frames within utterance, then category-stratified utterance bootstrap (2000) and two-sided t-test with Holm across ten fixed outcomes')
    write_json(out/'protocol_confirmation.json',protocol)
    rules=[]
    for stage in ['A','B','C']:rules+=json.loads((out/f'plan_{stage}.json').read_text())['rules']
    plan=dict(name='confirmation',rules=rules)
    secondary_plan=dict(name='confirmation',rules=[dict(layer=l,site=s,variants=['native','random','reverse']) for l,s in [(5,'attn'),(decision['selected_branch']['layer'],decision['selected_branch']['site'])]])
    write_json(out/'plan_confirmation.json',plan);write_json(out/'plan_secondary.json',secondary_plan)
    primary=['L5_attn_selective_excess','L11_attn_selective_excess','L11_mlp_selective_excess','L27_mlp_selective_excess',
        'coarse_minus_fine_selective_excess','middle_minus_fine_selective_excess','residual_L26_minus_L5_recovery','residual_L11_minus_L5_recovery',
        'second_history_L5_attn_selective_excess','second_history_selected_selective_excess']
    repo=Path(__file__).resolve().parents[1]
    paths={'cohort_primary':out/'confirmation_cohort.jsonl','cohort_secondary':out/'secondary_cohort.jsonl',
        'protocol':out/'protocol_confirmation.json','plan_primary':out/'plan_confirmation.json','plan_secondary':out/'plan_secondary.json',
        'basis':a.m1/'remote_export_final_v1/fit/basis.npz','run_script':repo/'scripts/run_m2.py','patch_engine':repo/'src/causal_patch.py',
        'analysis_script':repo/'scripts/analyze_m2.py','freeze_script':Path(__file__),'source_corpus':source,'exposed_prior':prior_path,'exposed_m1':m1_path}
    gate=dict(decision,confirmation_n=n,development_all_outcome_sd_upper=upper,required_n_all_primary_normal_approximation=required,
        cap_underpowered_for_minimum_effect=required>80,primary_outcomes=primary,selection_seed=2026111602,
        available_documents_by_category=counts,selected_category_counts={c:n//2 for c in ['referential_context','lexical_control']},
        cases_per_primary_history=len(case_list(plan)),cases_per_secondary_history=len(case_list(secondary_plan)),
        baseline_contexts=4*(n+20),intervention_passes=n*len(case_list(plan))+20*len(case_list(secondary_plan)),
        expected_seconds_per_primary_text_excluding_reference=10,reference_generation_limit_per_job_s=1200,
        implementation='Native full-position replacements; one dense matched random direction per cell; shared/specific off-manifold diagnostics separate',
        sensitivity='Exclude baseline boundary/EOS flagged cells as a separate descriptive analysis; retain all absolute outcomes; do not tune on confirmation',
        excluded_docs=len({r['source_document'] for r in prior}),source_results_hashes={str((Path(r['path']).parent/'results.json').relative_to(a.root)):r['artifacts']['results.json'] for r in lineage},
        **{k+'_sha256':digest(v.read_bytes()) for k,v in paths.items()})
    write_json(out/'confirmation_gate.json',gate)
    outcomes.to_csv(out/'development_planning_outcomes.csv',index=False)
    print(json.dumps({k:gate[k] for k in ['confirmation_n','required_n_all_primary_normal_approximation','cap_underpowered_for_minimum_effect','selected_branch','intervention_passes']}))


if __name__=='__main__':main()
