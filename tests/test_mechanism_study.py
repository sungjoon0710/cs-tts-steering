import numpy as np
import pytest
import torch
from src.mechanism_study import phase_pool,fit_structure,project_structure,holm,probability_context,probability_contrast,residual_decomposition
from src.generation_trace import prediction_differences


def test_phase_pool_excludes_anchor_and_weights_phases_explicitly():
    x=np.arange(21,dtype=np.float32)[None,:,None]
    out=phase_pool(x,np.arange(21)%7)
    assert out.shape==(1,7,1)
    assert out[0,0,0]==10.5 # j7/j14; j0 is not generation.
    assert out[0,1,0]==8
    assert phase_pool(x,np.arange(21)%7,False)[0,0,0]==7


def test_vectorized_probability_definitions_match_existing_reference_and_fix_eos():
    rng=np.random.default_rng(5)
    a=rng.normal(size=(14,156960)).astype('float32');b=a+rng.normal(scale=.2,size=a.shape).astype('float32')
    phases=np.arange(14)%7;targets=[128266+p*4096 for p in phases]
    with torch.inference_mode():
        x=probability_context(a,phases);y=probability_context(b,phases)
        rows=probability_contrast(x,y,targets,phases)
    reference=prediction_differences(a,b,targets)
    for r,s in zip(rows,reference):
        for key in ('jsd_nats','phase_jsd_nats','delta_target_logp','reference_entropy_nats'):
            assert r[key]==pytest.approx(s[key],abs=3e-6)
        assert r['reference_eos_probability']==pytest.approx(np.exp(s['reference_eos_probability']),rel=3e-6)
        assert 0<=r['reference_eos_probability']<=1
    assert max(r['jsd_nats'] for r in probability_contrast(x,x,targets,phases))<2e-6


def test_shared_and_style_residual_basis_is_fitted_only_on_supplied_development():
    rng=np.random.default_rng(7);x=rng.normal(scale=.02,size=(12,3,1,2,7,8))
    x[...,0]+=1
    for label in range(3):x[:,label,...,label+1]+=.7
    basis=fit_structure(x);before={k:v.copy() for k,v in basis.items()}
    out=project_structure(x+ rng.normal(scale=.02,size=x.shape),basis)
    assert out['shared_cosine'].mean()>.8 and out['specific_margin'].mean()>.5
    assert all(np.array_equal(v,before[k]) for k,v in basis.items())


def test_branch_projection_includes_rounding_remainder_and_excludes_anchor():
    rng=np.random.default_rng(6);a={'states':{s:rng.normal(size=(2,21,4)).astype('float32') for s in ('resid','attn','mlp')}}
    b={'states':{s:rng.normal(size=(2,21,4)).astype('float32') for s in ('resid','attn','mlp')}}
    rows=residual_decomposition(a,b,np.arange(21)%7)
    assert all(abs(sum(r[k] for k in ('incoming','attention','mlp','rounding'))-1)<1e-10 for r in rows)
    assert rows[0]['n_nonzero']==2


def test_holm_matches_known_stepdown_family():
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    with pytest.raises(ValueError):holm([np.nan])


def test_complete_five_style_runner_cpu_fixture(tmp_path):
    """Exercise collection, contrasts, masks and serialization, not just formulas."""
    import json
    from types import SimpleNamespace
    from test_generation_trace import TinyLM
    from scripts.run_m1 import inspect_history
    torch.set_num_threads(2)
    class Tokenizer:
        def convert_ids_to_tokens(self,x):return [str(i) for i in x]
        def decode(self,x):return 'TEST FIXTURE'
    model=SimpleNamespace(lm=TinyLM().eval(),device='cpu',tok=Tokenizer())
    model._build_prompt=lambda tok,text,voice,style:([2,3,4,5] if style else [])+[1,128257]
    protocol={'voice':'TEST','styles':{'default':'','anger':'ANGER','happy':'happy','sad':'sad','news':'news'}}
    p=tmp_path/'protocol.json';p.write_text(json.dumps(protocol))
    args=SimpleNamespace(output=tmp_path/'out',protocol=p,stage='development',snapshot=tmp_path/'model',codec_snapshot=tmp_path/'codec',basis=None)
    (args.output/'fixture').mkdir(parents=True)
    row={'id':'fixture','study_split':'development','category':'TEST','capture_exact':False,'numerical_repeat':False,'text':'test','text_sha256':'test'}
    historical={k:np.ones((2,8),np.float32) for k in ('anger','happy','sad','control_news')}
    codes=[128266+(j%7)*4096 for j in range(14)]
    done,deltas=inspect_history(model,row,0,{'test_only':True},codes,14,args,protocol,historical,None)
    assert done['completed'] and deltas.shape==(3,3,2,7,8)
    assert len(list((args.output/'fixture/h0').glob('*_predictions.csv')))==10
    assert (args.output/'fixture/h0/historical_alignment_positions.npz').exists()
    cached=inspect_history(model,row,0,{'test_only':True},codes,14,args,protocol,historical,None)
    assert cached['completed'] and cached['artifacts']==done['artifacts']
