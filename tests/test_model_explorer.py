"""CPU tests. Real artifacts are opt-in; no fixtures are published as experiments."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.model_explorer import Catalog, Explorer, bf16_round, identity, metrics, prediction_rows


def test_token_only_history_requires_normal_eos_and_complete_phases():
    from src.model_explorer import token_history_codes
    codes=[128266+4096*i for i in range(7)]
    assert token_history_codes({'generated_token_ids':codes+[128258]})==codes
    for ids in (codes, codes[:-1]+[128258], [codes[1]]+codes[1:]+[128258], codes+[128258,128258]):
        with pytest.raises(ValueError):token_history_codes({'generated_token_ids':ids})


def synthetic_m1_tree(root):
    """Tiny explicit TEST ONLY artifacts. Never a scientific study or GPU trace."""
    import csv
    from src.model_explorer import digest
    from src.generation_trace import prediction_positions
    def js(path,obj):path.write_text(json.dumps(obj),encoding='utf-8')
    def csvfile(path,rows):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    inputs=root/'inputs';inputs.mkdir(parents=True)
    protocol={'study':'TEST ONLY M1','language':'English','styles':{'default':'','test':'TEST ONLY'},'decoding':{},'primary_generation_rows':'j>=1'}
    js(inputs/'protocol.json',protocol)
    row={'id':'test01','text':'TEST ONLY','text_sha256':hashlib.sha256(b'TEST ONLY').hexdigest(),
        'study_split':'development','category':'test operational stratum'}
    (inputs/'cohort.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    model={'revision':'TEST','codec_revision':'TEST','layers':2,'hidden_size':8,'dtype':'bfloat16',
           'model_repo':'TEST ONLY','vendor_inference_sha256':'TEST'}
    js(inputs/'model_metadata.json',model)
    subject=root/'study/test01';source=subject/'h0';source.mkdir(parents=True)
    codes=[128266+4096*i for i in range(7)]
    js(subject/'history_0.json',{'generated_token_ids':codes+[128258], 'seed':1,
        'text_sha256':row['text_sha256'],'model_revision':'TEST','source':'TEST ONLY'})
    js(subject/'subject_complete.json',{'histories':1,'status':'complete'})
    for key in protocol['styles']:
        prompt=[128257] if key=='default' else [1,2,128257]
        js(source/(key+'.json'),{'prompt_ids':prompt,'history_tokens':codes,'prompt_length':len(prompt),
            'positions':prediction_positions(prompt,codes),'top_predictions':[{'j':j,'ids':[t],
                'probabilities':[.5],'tokens':['TEST ONLY']} for j,t in enumerate(codes)]})
        values=np.arange(2*2*8,dtype=np.float32).reshape(2,2,8)+(key=='test')
        np.savez_compressed(source/(key+'.npz'),resid=values,attn=values/4,mlp=values/2,
            prediction_rows=np.array([0,6]),prompt_embeddings=np.ones((len(prompt),8),np.float32),
            final_norm=np.ones((2,8),np.float32))
    prefix='test_minus_default'
    maps={site+'_'+metric:np.full((2,7),.25,np.float32) for site in ('resid','attn','mlp')
          for metric in ('normalized_l2','cosine_distance')}
    np.savez_compressed(source/(prefix+'_heatmaps.npz'),**maps)
    csvfile(source/(prefix+'_predictions.csv'),[{'j':j,'jsd_nats':.125,'argmax_changed':1,
        'reference_eos_probability':.01,'changed_eos_probability':.02} for j in range(7)])
    csvfile(source/'comparison_summary.csv',[{'reference':'default','condition':'test',
        'generation_JSD_nats':.125,'argmax_changed_fraction':1}])
    protocol_sha=digest(inputs/'protocol.json')
    js(source/'metadata.json',{'protocol_sha256':protocol_sha,'text_sha256':row['text_sha256'],
        'checkpoint':'TEST','codec':'TEST','n_positions':7,'voice':'TEST ONLY'})
    artifacts={p.name:digest(p) for p in source.iterdir()}
    js(source/'completion.json',{'completed':True,'protocol_sha256':protocol_sha,'artifacts':artifacts})
    return inputs/'model_metadata.json',source


@pytest.mark.parametrize('exact',[False,True])
def test_m1_export_preserves_scalars_and_explicit_token_only_availability(tmp_path,exact):
    from scripts.export_m1_explorer import export,parser
    from src.model_explorer import digest
    source=tmp_path/'source';model,history=synthetic_m1_tree(source)
    output=tmp_path/'export';archive=tmp_path/'pack.zip'
    cli=['--root',str(source),'--output',str(output),'--model-metadata',str(model),
         '--vector-row-limit','1','--archive',str(archive),'--max-compressed-mib','0']
    if exact:cli+=['--exact-ids','test01']
    report=export(parser().parse_args(cli))
    assert report['exported_contexts']==2 and report['exported_histories']==1
    assert report['within_target_compressed_budget'] is False
    assert report['archive_bytes']==archive.stat().st_size>0
    target=output/'test01/h0'
    for name in ('test_minus_default_heatmaps.npz','test_minus_default_predictions.csv'):
        assert digest(history/name)==digest(target/name)
    with np.load(target/'test.npz') as z:
        assert z['prediction_rows'].tolist()==([6] if exact else [])
        if exact:
            with np.load(history/'test.npz') as raw:np.testing.assert_array_equal(z['resid'],raw['resid'][:,1:])
    cat=Catalog(tmp_path/'catalog.sqlite');cat.register_spec(target/'manifest.json');cat.verify_sources();cat.close()
    e=Explorer(tmp_path/'catalog.sqlite');data=e.catalog_data();pair=data['comparisons'][0]
    context=e.describe(pair['a']);hid=context['history_id']
    assert context['recording'] is None and not e.query('SELECT * FROM recording')
    assert e.decoder_info(hid)['available'] is False and e.decoder_info(hid)['stages']==[]
    assert e.waveform(hid)['available'] is False
    with pytest.raises(ValueError,match='Token-only'):e.decoder(hid,'z_q',0)
    assert e.vectors(pair['a'],pair['b'],1,6,'resid')['available'] is exact
    assert e.embeddings(pair['a'],pair['b'],0,0)['available'] is exact
    assert e.predictions(pair['a'],pair['b'],6)['a']['ids']==[128266+4096*6]
    assert e.comparison(pair['a'],pair['b'])['summary']['jsd_mean_nats']==.125
    manifest=json.loads((target/'manifest.json').read_text())
    manifest['contexts'][0]['metadata']['positions'][0]['processed_phase']=0
    bad=target/'bad_manifest.json';bad.write_text(json.dumps(manifest),encoding='utf-8')
    cat=Catalog(tmp_path/'bad.sqlite')
    with pytest.raises(ValueError,match='phase'):cat.register_spec(bad)
    cat.close()


def test_condition_identity_uses_exact_semantics_not_display_name():
    a = {'style':'speaking with warmth', 'speaker':'Amit', 'model_revision':'r1'}
    b = {**a, 'style':'speaking with warmth and empathy'}
    assert identity('condition', a) == identity('condition', dict(reversed(list(a.items()))))
    assert identity('condition', a) != identity('condition', b)


def test_bfloat_rounding_ties_and_residual_addition():
    # Between 1 and the next BF16 value, ties-to-even rounds down; next tie up.
    x = np.array([1+1/256, 1+3/256, -1-1/256], np.float32)
    np.testing.assert_array_equal(bf16_round(x), [1, 1+2/128, -1])
    with pytest.raises(ValueError, match='Nonfinite'):
        bf16_round([np.nan])
    assert metrics([1,0],[0,1])['cosine_similarity'] == 0


def test_eos_legacy_conversion_preserves_raw_data(tmp_path):
    p = tmp_path/'p.csv'
    p.write_text('reference_eos_probability,changed_eos_probability\n-2,-3\n', encoding='utf-8')
    before = p.read_bytes()
    row = prediction_rows(p)[0]
    assert row['reference_eos_probability_raw'] == -2
    assert row['reference_eos_probability'] == pytest.approx(np.exp(-2))
    assert row['changed_eos_logp'] == -3 and 'legacy' in row['eos_encoding']
    assert before == p.read_bytes()


def test_generic_manifest_accepts_arbitrary_metadata_and_rejects_changed_artifact(tmp_path):
    # Explicitly synthetic unit fixtures, never registered in the research catalog.
    from src.model_explorer import digest
    def file(name,content):
        p=tmp_path/name;p.write_text(json.dumps(content),encoding='utf-8')
        return {'path':name,'sha256':digest(p)}
    ids=[128266+4096*i for i in range(7)]
    np.savez_compressed(tmp_path/'hidden.npz',resid=np.zeros((2,7,8),np.float32),
        attn=np.zeros((2,7,8),np.float32),mlp=np.zeros((2,7,8),np.float32),prediction_rows=np.arange(7),
        prompt_embeddings=np.zeros((1,8),np.float32),final_norm=np.zeros((7,8),np.float32))
    np.savez_compressed(tmp_path/'decoder.npz',codebook_0_ids=np.zeros((1,1),np.int64),
        codebook_1_ids=np.zeros((1,2),np.int64),codebook_2_ids=np.zeros((1,4),np.int64))
    (tmp_path/'audio.wav').write_bytes(b'TEST ONLY; not experimental audio')
    reference=lambda name:{'path':name,'sha256':digest(tmp_path/name)}
    metadata={'style':'test-only descriptive instruction','speaker':'TEST','language':'TEST',
        'prompt_template':{'name':'test fixture'},'settings':{'test_only':True}}
    positions=[{'j':j,'absolute_position':j,'predicted_phase':j,'processed_phase':j-1,'frame':0,
                'input_token':128257 if j==0 else ids[j-1],'target_token':ids[j]} for j in range(7)]
    spec={'schema_version':1,'experiment':{'name':'TEST ONLY','dataset':'unit fixture','model':{
        'revision':'test','codec_revision':'test','layers':2,'hidden_size':8,'dtype':'bfloat16'}},
        'utterance':{'id':'fixture','text':'unit fixture','metadata':{'test_only':True}},
        'conditions':{'arbitrary-key':{'label':'Not an emotion preset','metadata':metadata}},
        'histories':{'codes':{'label':'test history','recording':{'audio':reference('audio.wav'),'metadata':{'test_only':True}},
            'decoder_npz':reference('decoder.npz'),'decoder_json':file('decoder.json',{})}},
        'contexts':[{'key':'replay','condition':'arbitrary-key','history':'codes','metadata':{
            'prompt_ids':[128257],'history_tokens':ids,'positions':positions},'arrays':reference('hidden.npz'),
            'predictions':file('top.json',{'top_predictions':[]}), 'capture_levels':{'test_only':True}}]}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(spec),encoding='utf-8')
    cat=Catalog(tmp_path/'catalog.sqlite');result=cat.register_spec(p);assert result['contexts']==1
    cat.verify_sources();cat.close()
    e=Explorer(tmp_path/'catalog.sqlite');ctx=e.context(e.catalog_data()['contexts'][0]['id'])
    assert e._vectors(ctx)['resid'].shape==(2,7,8)
    (tmp_path/'hidden.npz').write_bytes(b'changed test fixture')
    cat=Catalog(tmp_path/'catalog.sqlite')
    with pytest.raises(ValueError,match='changed'):
        cat.verify_sources()
    cat.close()


@pytest.fixture(scope='module')
def registered(tmp_path_factory):
    supplied = os.environ.get('SPEECHLAB_TRACE_ROOT')
    if not supplied:
        pytest.skip('Set SPEECHLAB_TRACE_ROOT to the existing generation_introspection run; no inference needed')
    root = Path(supplied)
    path = tmp_path_factory.mktemp('explorer')/'catalog.sqlite'
    cat = Catalog(path)
    for rel in ('remote_results/trace', 'remote_results_c04/trace'):
        cat.register_legacy(root/rel,'real-artifact CPU test','existing authored texts','en')
    cat.close()
    return path


def first_pair(e):
    return e.catalog_data()['comparisons'][0]


def test_existing_comparisons_and_phase_alignment_are_unchanged(registered):
    e=Explorer(registered)
    for p in e.catalog_data()['comparisons']:
        r=e.comparison(p['a'],p['b'])
        assert np.mean([x['jsd_nats'] for x in r['predictions']]) == pytest.approx(p['metadata']['jsd_mean_nats'],abs=1e-14)
        assert r['positions_a'][175]['predicted_phase']==0
        assert r['positions_a'][175]['processed_phase']==6
        assert r['positions_a'][175]['frame']==25
        assert all(0<=x['reference_eos_probability']<=1 and 0<=x['changed_eos_probability']<=1 for x in r['predictions'])


def test_all_layers_saved_vectors_and_verified_residual_arithmetic(registered):
    e=Explorer(registered)
    n=0
    for row in e.catalog_data()['contexts']:
        ctx=e.context(row['id'])
        for layer in range(28):
            for j in ctx['capture']['metadata']['full_vector_positions']:
                v,why=e._vector(ctx,layer,j,'after_attention')
                if layer==0 and j!=0:
                    assert v is None
                else:
                    assert v.shape==(3072,) and 'bit-exact' in why
                    n+=1
    assert n==8*(27*14+1)


def test_missing_tensor_positions_and_normalization_inputs(registered):
    e=Explorer(registered);p=first_pair(e)
    v=e.vectors(p['a'],p['b'],14,175,'resid')
    assert not v['available'] and 'a' not in v
    assert e.vectors(p['a'],p['b'],14,0,'resid')['available']
    assert not e.vectors(p['a'],p['b'],14,0,'attention_norm')['available']
    with pytest.raises(ValueError,match='Unknown tensor site'):
        e.vectors(p['a'],p['b'],14,175,'invented')


def test_pinned_layers_share_one_projection_and_original_labels(registered):
    e=Explorer(registered);p=first_pair(e)
    r=e.geometry(p['a'],p['b'],[5,26],'resid')
    assert len(r['points'])==56 and {x['layer'] for x in r['points']}=={5,26}
    assert {x['condition'] for x in r['points']}=={'A','B'}
    assert abs(np.mean([x['x'] for x in r['points']]))<1e-12
    assert 0<sum(r['explained_variance_ratio'])<=1
    assert r['fit_id']==e.geometry(p['a'],p['b'],[5,26],'resid')['fit_id']


def test_snac_vectors_partial_projection_and_vocos_spectrum(registered):
    e=Explorer(registered);p=first_pair(e);hid=e.context(p['a'])['history_id']
    z=e.decoder(hid,'z_q',25)
    assert z['shape'][0]==768 and z['checks']['saved_first_16_channel_sum_exact']
    assert z['checks']['reconstructed_channels']==16
    assert e.decoder(hid,'snac.quantizer.0.out_proj.lookup.channels',25)['shape'][0]==8
    assert e.decoder(hid,'snac.quantizer.1.out_proj.channels',25)['shape'][0]==16
    for stage in e.decoder_info(hid)['stages']:
        d=e.decoder(hid,stage['key'],25)
        assert d['matrix'] and d['profile'] and len(d['vector'])==d['shape'][0]
    assert e.decoder(hid,'vocos.predicted_stft_phase',25)['shape'][0]==257


def test_metadata_only_registration_does_not_create_traces(registered):
    cat=Catalog(registered)
    before=Explorer(registered).catalog_data()
    meta={**before['conditions'][0]['metadata'],'style':'TEST ONLY · no speech or trace produced'}
    cid=cat.register_condition(before['experiments'][0]['id'],'Metadata-only test',meta)
    cat.db.commit();cat.close()
    after=Explorer(registered).catalog_data()
    assert any(x['id']==cid for x in after['conditions'])
    assert len(before['contexts'])==len(after['contexts'])==8
    assert not any(x['condition_id']==cid for x in after['contexts'])


def test_unknown_and_unrecorded_comparisons_are_rejected(registered):
    e=Explorer(registered);p=first_pair(e)
    with pytest.raises(ValueError,match='Unknown context'):
        e.compatible('no_such_context',p['b'])
    a=e.context(p['a'])
    other=next(x for x in e.catalog_data()['contexts'] if x['utterance_id']!=a['utterance_id'])
    with pytest.raises(ValueError,match='Different utterances'):
        e.comparison(p['a'],other['id'])
    with pytest.raises(ValueError,match='No stored ordered'):
        e.comparison(p['b'],p['a'])
    # Exact selected vectors can still be compared in reverse; probabilities
    # and all-position normalized L2 are NOT reconstructed from insufficient data.
    assert e.vectors(p['b'],p['a'],14,0,'resid')['available']


def test_immutable_source_inventory_still_verifies(registered):
    cat=Catalog(registered)
    reports=cat.verify_sources();cat.close()
    assert len(reports)==2 and sum(r['verified_artifacts'] for r in reports)==60
