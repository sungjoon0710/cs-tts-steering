"""CPU-only synthetic checks; no model, GPU or experimental artifacts required."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_m1 import (CODEC_REVISION, MODEL_REVISION, CONTRASTS, SITES, STYLES,
                               check_artifacts, check_pair, check_positions, check_predictions,
                               cohort_checks, full_codes, gate_checks, sha, verify, verify_history, reference_failure)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positions(prompt, codes):
    return [dict(j=j, absolute_position=len(prompt)-1+j, processed_phase=-1 if j == 0 else (j-1)%7,
                 predicted_phase=j%7, frame=j//7, input_token=prompt[-1] if j == 0 else codes[j-1], target_token=t)
            for j, t in enumerate(codes)]


def predictions(codes):
    return [dict(j=j, predicted_phase=j%7, target_token=t, jsd_nats=.02, phase_jsd_nats=.03,
                 reference_target_logp=-2., changed_target_logp=-1.9, delta_target_logp=.1,
                 reference_phase_mass=.9, changed_phase_mass=.9, reference_eos_probability=.01,
                 changed_eos_probability=.01, reference_eos_logp=np.log(.01), changed_eos_logp=np.log(.01),
                 reference_entropy_nats=2., changed_entropy_nats=2.1, argmax_changed=0)
            for j, t in enumerate(codes)]


def cohort(root):
    rows = []
    for role, total, second in [('development', 30, 10), ('confirmation', 90, 30)]:
        for j in range(total):
            text = f'{role} sample {j}'
            rows.append(dict(id=f'{role}_{j}', study_split=role, source_document=f'{role}_doc_{j}',
                             text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             category=('emotion_marked', 'referential_context', 'lexical_control')[j%3], second_history=j<second))
    path = root/'inputs/cohort.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    prior = root/'inputs/prior_cohort.jsonl'
    prior.write_text(json.dumps(dict(source_document='old', text_SHA256='old'))+'\n', encoding='utf-8')
    write(root/'inputs/selection.json', dict(cohort_sha256=sha(path), prior_manifest_sha256=sha(prior)))
    return rows


def one_history(root, k=0):
    row = dict(id='sample', text_sha256='text', study_split='development', category='emotion_marked')
    protocol = dict(voice='Amit', min_frames=1, max_frames=2)
    codes = [128266+(j%7)*4096 for j in range(14)]
    history = dict(generated_token_ids=codes+[128258], prompt_ids=[4,128257], text_sha256='text',
                   model_revision=MODEL_REVISION, history_index=k)
    subject = root/'study/sample'
    directory = subject/f'h{k}'
    directory.mkdir(parents=True)
    write(subject/f'history_{k}.json', history)
    checks = dict(hooks_removed=True, fresh_uncached=True, residual_reconstruction_exact=True, no_hook_logits_equal=None)
    lengths = {}
    for style in STYLES:
        prompt = ([4] if style == 'default' else [4,5,6,7,8])+[128257]
        lengths[style] = len(prompt)
        write(directory/(style+'.json'), dict(prompt_ids=prompt, prompt_length=len(prompt), prompt_token_pieces=['p']*len(prompt),
              history_tokens=codes, positions=positions(prompt,codes), evidence='observational_replay', checks=checks,
              top_predictions=[dict(j=j, ids=[1,2,3,4,5], probabilities=[.1]*5, tokens=['t']*5) for j in range(14)]))
        np.savez_compressed(directory/(style+'.npz'), **{s:np.empty((28,0,3072)) for s in SITES},
                            prediction_rows=np.array([],dtype=int), prompt_embeddings=np.ones((len(prompt),3072)), final_norm=np.empty((0,3072)))
    layer, probability, summaries, decomposition = [], [], [], []
    for a,b in CONTRASTS:
        prefix = b+'_minus_'+a
        table(directory/(prefix+'_predictions.csv'),predictions(codes))
        np.savez_compressed(directory/(prefix+'_heatmaps.npz'), **{s+'_'+metric:np.ones((28,14),np.float32)*.1 for s in SITES for metric in ('cosine_distance','normalized_l2')})
        summaries.append(dict(reference=a,condition=b,positions=14,generation_JSD_nats=.02))
        for p in range(7):
            count = 1 if p == 0 else 2
            probability.append(dict(reference=a,condition=b,phase=p,n_positions=count,jsd_nats=.02,phase_jsd_nats=.03,
                                    delta_target_logp=.1,argmax_changed=0,reference_entropy_nats=2.,changed_entropy_nats=2.1))
            for site in SITES:
                for layer_id in range(28):
                    layer.append(dict(reference=a,condition=b,site=site,layer=layer_id,phase=p,n_positions=count,cosine_distance=.1,normalized_l2=.1))
    for style in STYLES[1:4]:
        for l in range(28):
            for p in range(7):
                decomposition.append(dict(condition=style,layer=l,phase=p,incoming=.2,attention=.3,mlp=.4,rounding=.1,n_nonzero=1))
    for name, rows in [('layer_phase_metrics',layer),('probability_phase_metrics',probability),('comparison_summary',summaries),('residual_decomposition',decomposition)]:
        table(directory/(name+'.csv'),rows)
    np.savez_compressed(directory/'geometry_metrics.npz', historical_news=np.zeros((3,28,7)),historical_default=np.zeros((3,28,7)))
    np.savez_compressed(directory/'historical_alignment_positions.npz', news=np.zeros((3,28,14)),default=np.zeros((3,28,14)))
    write(directory/'metadata.json',dict(**row,history_index=k,history_provenance=history,n_positions=14,generation_rows='j>=1',phase_weighting='equal',
          protocol_sha256='protocol',checkpoint=MODEL_REVISION,codec=CODEC_REVISION,voice='Amit',decoder_replayed=False,style_prompt_lengths=lengths,
          checks={s:checks for s in STYLES},tensor_rows=[],basis_sha256=None))
    write(directory/'completion.json',dict(completed=True,id='sample',history=k,protocol_sha256='protocol',positions=14,
          artifacts={p.name:sha(p) for p in directory.iterdir()}))
    return row, protocol, directory


def test_position_semantics_and_code_validation():
    codes=[128266+(j%7)*4096 for j in range(14)]
    assert full_codes(dict(generated_token_ids=codes+[128258])) == codes
    p=positions([1,128257],codes)
    check_positions(p,[1,128257],codes)
    p[1]['processed_phase']=1
    with pytest.raises(ValueError,match='association'):
        check_positions(p,[1,128257],codes)
    with pytest.raises(ValueError,match='phase/range'):
        full_codes(dict(generated_token_ids=[128266]*14+[128258]))


def test_probability_eos_regression_and_target_alignment():
    codes=[128266+(j%7)*4096 for j in range(14)]
    rows=predictions(codes)
    check_predictions(rows,codes)
    rows[0]['reference_eos_probability']=np.log(.01)
    with pytest.raises(ValueError,match='actual probability'):
        check_predictions(rows,codes)


def test_complete_history_and_missing_tensor_are_distinct(tmp_path):
    row,protocol,directory=one_history(tmp_path)
    result=verify_history(tmp_path,row,0,protocol,'protocol',None)
    assert result['status']=='fully_verified' and result['positions']==14
    (directory/'anger.npz').unlink()
    with pytest.raises(ValueError,match='missing'):
        verify_history(tmp_path,row,0,protocol,'protocol',None)
    result=verify_history(tmp_path,row,0,protocol,'protocol',None,True)
    assert result['status']=='available_files_verified' and result['omitted_tensors']=='anger.npz'
    (directory/'geometry_metrics.npz').unlink()
    with pytest.raises(ValueError,match='missing'):
        verify_history(tmp_path,row,0,protocol,'protocol',None,True)


def test_artifact_change_is_not_an_allowed_omission(tmp_path):
    row,protocol,directory=one_history(tmp_path)
    (directory/'anger.json').write_text('{}')
    with pytest.raises(ValueError,match='hash differs'):
        verify_history(tmp_path,row,0,protocol,'protocol',None,True)


def test_cohort_exposure_and_manifest_hashes(tmp_path):
    rows=cohort(tmp_path)
    actual,report,warnings=cohort_checks(tmp_path)
    assert len(actual)==120 and report['prior_exposure_verified'] and not warnings
    prior=tmp_path/'inputs/prior_cohort.jsonl'
    prior.write_text(json.dumps(dict(source_document=rows[-1]['source_document'],text_SHA256='other'))+'\n')
    selection=json.loads((tmp_path/'inputs/selection.json').read_text())
    selection['prior_manifest_sha256']=sha(prior)
    write(tmp_path/'inputs/selection.json',selection)
    with pytest.raises(ValueError,match='overlaps'):
        cohort_checks(tmp_path)


def test_interim_report_and_no_overwrite(tmp_path):
    root=tmp_path/'source'
    cohort(root)
    write(root/'inputs/protocol.json',dict(voice='Amit',styles=dict(zip(STYLES,('', 'ANGER','happy','sad','news')))))
    source_hash=sha(root/'inputs/cohort.jsonl')
    output=tmp_path/'interim'
    report=verify(root,output,allow_partial=True)
    assert report['status']=='interim_available_files_verified'
    assert report['history_status_counts']=={'not_completed_interim':160}
    assert sha(root/'inputs/cohort.jsonl')==source_hash
    with pytest.raises(ValueError,match='must be new'):
        verify(root,output,allow_partial=True)
    assert (output/'verified_manifest.csv').is_file()


def test_gate_rejects_confirmation_training(tmp_path):
    rows=cohort(tmp_path)
    write(tmp_path/'inputs/protocol.json',{})
    write(tmp_path/'fit/confirmation_gate.json',dict(cohort_sha256=sha(tmp_path/'inputs/cohort.jsonl'),
          protocol_sha256=sha(tmp_path/'inputs/protocol.json'),development_ids=[rows[-1]['id']]))
    with pytest.raises(ValueError,match='non-development'):
        gate_checks(tmp_path,tmp_path/'inputs/cohort.jsonl',tmp_path/'inputs/protocol.json')


def test_duplicate_history_is_not_independent_evidence(tmp_path):
    row,_,_=one_history(tmp_path,0)
    one_history(tmp_path,1)
    subject=tmp_path/'study/sample'
    write(subject/'history_comparison.json',dict(inspected_tokens=14,identical_inspected_prefix=True,token_agreement_fraction=1.))
    np.savez_compressed(subject/'history_robustness.npz',cosine=np.ones((3,3,28,7)))
    assert check_pair(tmp_path,row)['independent_history_evidence'] is False
    (subject/'history_robustness.npz').unlink()
    with pytest.raises(FileNotFoundError):
        check_pair(tmp_path,row)


def test_capped_reference_is_missing_not_zero_or_generic_failure(tmp_path):
    from src.tts_settings import token_budget
    row={'id':'capped','text':'x'*103,'text_sha256':hashlib.sha256(('x'*103).encode()).hexdigest(),'second_history':False}
    subject=tmp_path/'study/capped';cap=token_budget(row['text'])
    history=dict(generated_token_ids=[128266+(j%7)*4096 for j in range(cap)],prompt_ids=[4,128257],
                 source='generated_missing_reference_codes_only',text_sha256=row['text_sha256'],model_revision=MODEL_REVISION,
                 history_index=0,seed=1234,hit_token_cap=True,max_new_tokens=cap,audio_generated=False,decoder_executed=False)
    failure=dict(id='capped',type='ValueError',message='Reference continuation lacks normal EOS or has trailing/invalid codes')
    write(subject/'history_0.json',history);write(subject/'failure.json',failure)
    protocol=dict(history_seed=1234,secondary_history_seed=5678)
    assert reference_failure(tmp_path,row,protocol)['status']=='reference_token_cap'
    history['seed']=999;write(subject/'history_0.json',history)
    with pytest.raises(ValueError,match='seed differs'):reference_failure(tmp_path,row,protocol)
    write(subject/'failure.json',{**failure,'message':'Hash mismatch'})
    with pytest.raises(ValueError,match='technical failure'):reference_failure(tmp_path,row,protocol)
