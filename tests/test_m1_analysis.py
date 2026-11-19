"""Synthetic CPU fixtures only; no fixture is an executed research result."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analyze_m1 import (bootstrap_mean,bootstrap_weights,build,collect,digest,
                               primary_tests,projection_missingness,summarize_cells,tie_description,verify_gate,verify_transfer_receipt)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def write_json(path,obj):
    path.write_text(json.dumps(obj),encoding='utf-8')


def test_stratified_bootstrap_keeps_utterance_rows_and_missing_cells():
    strata=['a','a','b','b'];weights=bootstrap_weights(strata,100,7)
    np.testing.assert_array_equal(weights[:,:2].sum(1),2)
    np.testing.assert_array_equal(weights[:,2:].sum(1),2)
    x=np.array([[1,1,np.nan],[2,2,np.nan],[3,3,np.nan],[4,4,np.nan]],float)
    b=bootstrap_mean(x,strata,100,7,True)
    assert b['mean'][0]==2.5 and np.isnan(b['mean'][2]) and b['n'][2]==0
    assert b['max_deviation_critical']>0
    # Two duplicated features share exactly the same dependence-preserving band.
    assert b['simultaneous_lower'][0]==b['simultaneous_lower'][1]


def test_fixed_family_has_no_zero_imputation_and_reports_opposite_sign():
    rows=[{'study_split':'confirmation','category':'a','x':-1-float(i)/10,'missing':np.nan} for i in range(8)]
    out=primary_tests(rows,['x','missing'],100,3,8,0)
    assert out[0]['mean']<0 and out[0]['reject_at_05']
    assert out[1]['n']==0 and np.isnan(out[1]['mean']) and out[1]['p_raw'] is None
    assert out[1]['p_holm']==1


def test_summary_rejects_listener_or_token_pseudoreplication():
    rows=[{'id':'same','category':'a','phase':0,'x':1},{'id':'same','category':'a','phase':0,'x':2}]
    with pytest.raises(ValueError,match='Repeated utterance'):
        summarize_cells(rows,('phase',),'x',100,4)


def test_undefined_projection_and_exact_selection_ties():
    basis={'shared':np.ones((3,28,7,2)), 'specific':np.ones((3,3,28,7,2))}
    g={'shared_cosine':np.full((3,3,28,7),.2),
       'specific_cosine':np.full((3,3,28,7),.1), 'specific_margin':np.full((3,3,28,7),.1)}
    g['shared_cosine'][0,0,0,0]=0;g['specific_cosine'][0,0,0,0]=np.nan
    basis['shared'][1,1,1]=0
    cleaned,audit=projection_missingness(g,basis)
    assert audit['zero_input_candidate_cells']==1
    assert np.isnan(cleaned['shared_cosine'][0,0,0,0])
    assert np.isnan(cleaned['shared_cosine'][:,1,1,1]).all()
    assert g['shared_cosine'][0,0,0,0]==0  # source copy is preserved
    tie=tie_description(np.array([[1.,2.],[2.,1.]]),0,1)
    assert tie['exact_tie_flat_indices']==[1,2] and tie['matches_first_flat_rule']


def test_transfer_receipt_rejects_changed_or_escaping_artifacts(tmp_path):
    root=tmp_path/'export';root.mkdir();artifact=root/'small.csv';artifact.write_bytes(b'TEST ONLY')
    write_json(root/'transfer_manifest.json',{'sha256':{'small.csv':digest(artifact.read_bytes())}})
    assert verify_transfer_receipt(root)['verified_files']==1
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'):verify_transfer_receipt(root)
    write_json(root/'transfer_manifest.json',{'sha256':{'../outside.csv':'irrelevant'}})
    with pytest.raises(ValueError,match='Unsafe or missing'):verify_transfer_receipt(root)


@pytest.fixture
def synthetic_study(tmp_path):
    root=tmp_path/'TEST_ONLY_STUDY';(root/'inputs').mkdir(parents=True);(root/'fit').mkdir()
    categories=['emotion_marked','referential_context','lexical_control'];cohort=[]
    for i in range(9):
        split='development' if i<3 else 'confirmation'
        cohort.append({'id':f'test{i}','study_split':split,'category':categories[i%3],
                       'text_sha256':f'test_text_{i}','source_document':f'TEST_ONLY_DOC_{i}',
                       'second_history':i in (3,4,5)})
    (root/'inputs/cohort.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cohort))
    outcomes=['historical_news_anchor_alignment','shared_alignment_resid','specificity_margin_resid',
              'shared_alignment_attn','specificity_margin_attn','shared_alignment_mlp','specificity_margin_mlp',
              'residual_magnitude_L26_minus_L5','phase0_minus_other_phases_JSD','history_direction_cosine']
    protocol={'study':'TEST ONLY SYNTHETIC','voice':'TEST_VOICE','bootstrap':{'replicates':2000},'seed_for_analysis':20261116,'primary_outcomes':outcomes}
    write_json(root/'inputs/protocol.json',protocol)
    np.savez_compressed(root/'fit/basis.npz',shared=np.ones((3,28,7,2)),specific=np.ones((3,3,28,7,2)))
    write_json(root/'fit/M2_shortlist.json',{'shared_structure':{'site':'resid','layer':5,'development_mean':.6},
        'style_specific_structure':{'site':'mlp','layer':26,'development_mean':.3},'prediction_phase':{'phase':0}})
    gate={'development_ids':[r['id'] for r in cohort[:3]],'basis_sha256':digest((root/'fit/basis.npz').read_bytes()),
          'shortlist_sha256':digest((root/'fit/M2_shortlist.json').read_bytes()),
          'cohort_sha256':digest((root/'inputs/cohort.jsonl').read_bytes()),
          'protocol_sha256':digest((root/'inputs/protocol.json').read_bytes()),'fitted_on_confirmation':False}
    write_json(root/'fit/confirmation_gate.json',gate)
    rng=np.random.default_rng(21)
    for index,r in enumerate(cohort):
        subject=root/'study'/r['id'];subject.mkdir(parents=True)
        histories=2 if r['second_history'] else 1
        for h in range(histories):
            dest=subject/f'h{h}';dest.mkdir()
            g={'historical_news':rng.normal(.2+index/100,.01,(3,28,7)),
               'historical_default':rng.normal(.4+index/100,.01,(3,28,7)),
               'shared_cosine':rng.normal(.5+index/100,.01,(3,3,28,7)),
               'specific_margin':rng.normal(.15+index/100,.01,(3,3,28,7))}
            np.savez_compressed(dest/'geometry_metrics.npz',**g)
            if r['study_split']=='development' and h==0:np.savez_compressed(root/'fit'/(r['id']+'_geometry.npz'),**g)
            np.savez_compressed(dest/'historical_alignment_positions.npz',
                news=rng.normal(.2,.02,(3,28,14)),default=rng.normal(.3,.02,(3,28,14)))
            write_json(dest/'metadata.json',{'protocol_sha256':gate['protocol_sha256'],
                'basis_sha256':gate['basis_sha256'] if r['study_split']=='confirmation' else None,
                'id':r['id'],'study_split':r['study_split'],'category':r['category'],'history_index':h,
                'voice':'TEST_VOICE','n_positions':14,'checkpoint':'TEST_CHECKPOINT','codec':'TEST_CODEC',
                'text_sha256':r['text_sha256'],'history_provenance':{'generated_token_ids':list(range(15))}})
            layers=[];prob=[];components=[]
            for style in ('anger','happy','sad'):
                for phase in range(7):
                    for layer in (5,26):layers.append({'reference':'news','site':'resid','condition':style,'layer':layer,'phase':phase,'normalized_l2':.1+index*.002+layer*.001})
                    for ref in ('news','default'):
                        prob.append({'reference':ref,'condition':style,'phase':phase,'n_positions':1 if phase==0 else 2,
                            'jsd_nats':.01+index*.0001+phase*.0001+h*.00002,'phase_jsd_nats':.009+index*.0001,
                            'delta_target_logp':-.01+index*.0002,'argmax_changed':.2+index*.001,
                            'reference_entropy_nats':3.,'changed_entropy_nats':3.01})
                    for layer in range(28):components.append({'condition':style,'layer':layer,'phase':phase,'incoming':.5,'attention':.3,'mlp':.18,'rounding':.02,'n_nonzero':1})
                write_csv(dest/(style+'_minus_news_predictions.csv'),[{'j':j,'jsd_nats':.01+index*.001+j*.00001} for j in range(14)])
            write_csv(dest/'layer_phase_metrics.csv',layers);write_csv(dest/'probability_phase_metrics.csv',prob);write_csv(dest/'residual_decomposition.csv',components)
            artifacts={p.name:digest(p.read_bytes()) for p in dest.iterdir() if p.is_file()}
            write_json(dest/'completion.json',{'protocol_sha256':gate['protocol_sha256'],'positions':14,'artifacts':artifacts})
        if histories==2:
            np.savez_compressed(subject/'history_robustness.npz',cosine=np.full((3,3,28,7),.8+index*.005))
            write_json(subject/'history_comparison.json',{'identical_inspected_prefix':False})
        write_json(subject/'subject_complete.json',{'histories':histories})
    dev=[dict(np.load(root/'fit'/(r['id']+'_geometry.npz'))) for r in cohort[:3]]
    shortlist={}
    for name,key in [('shared_structure','shared_cosine'),('style_specific_structure','specific_margin')]:
        score=np.nanmean(np.stack([g[key] for g in dev]),axis=(0,1,4))
        site,layer=np.unravel_index(np.nanargmax(score),score.shape)
        shortlist[name]={'site':('resid','attn','mlp')[site],'layer':int(layer),'development_mean':float(score[site,layer])}
    shortlist['prediction_phase']={'phase':6}
    write_json(root/'fit/M2_shortlist.json',shortlist)
    gate['shortlist_sha256']=digest((root/'fit/M2_shortlist.json').read_bytes())
    write_json(root/'fit/confirmation_gate.json',gate)
    return root


def test_gate_rejects_changes_and_unattempted_confirmation(synthetic_study):
    root=synthetic_study;gate,cohort=verify_gate(root)
    path=root/'study/test8/subject_complete.json';path.unlink()
    with pytest.raises(ValueError,match='Unattempted'):
        collect(root,gate,cohort)
    (root/'fit/M2_shortlist.json').write_text('{}')
    with pytest.raises(ValueError,match='shortlist'):
        verify_gate(root)


def test_complete_cpu_analysis_fixture_and_no_confirmation_fit(synthetic_study,tmp_path):
    root=synthetic_study;before=(root/'fit/basis.npz').read_bytes()
    args=SimpleNamespace(root=root,output=tmp_path/'TEST_ONLY_OUTPUT',allow_partial=True,
                         allow_missing_tensors=False,bootstrap=30)
    build(args)
    assert len(list(args.output.glob('0*.png')))==5
    assert (root/'fit/basis.npz').read_bytes()==before
    rows=list(csv.DictReader((args.output/'primary_confirmation_tests.csv').open()))
    assert len(rows)==10 and all(int(r['n']) in (3,6) for r in rows)
    manifest=json.loads((args.output/'analysis_manifest.json').read_text())
    assert manifest['partial_results'] and not manifest['basis_refitted']
