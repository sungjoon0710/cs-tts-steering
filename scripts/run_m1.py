"""Bounded M1 extraction on one GPU. No steering, training, ASR or decoder replay."""
import argparse,csv,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.inspect_representations import load_model
from src.evaluation.common import digest,read_jsonl,write_json
from src.generation_trace import trace_context,prediction_positions,state_differences
from src.representation_probe import code_phases
from src.tts_settings import token_budget
from src.mechanism_study import STYLES,SITES,CONTRASTS,phase_pool,cosine,probability_context,probability_contrast,project_structure


def savecsv(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def valid_codes(record):
    ids=record['generated_token_ids'];phases=code_phases(ids);n=sum(p>=0 for p in phases)
    if not n or n%7 or phases[:n]!=[i%7 for i in range(n)] or any(p>=0 for p in phases[n:]):
        raise ValueError('Invalid/incomplete/non-contiguous speech-code frames')
    if len(ids)!=n+1 or ids[-1] not in (128258,128001):
        raise ValueError('Reference continuation lacks normal EOS or has trailing/invalid codes')
    return ids[:n]


def prepare_history(model,row,index,args,protocol,output):
    target=output/f'history_{index}.json'
    if target.exists():
        rec=json.loads(target.read_text())
        if rec['text_sha256']!=row['text_sha256'] or rec['model_revision']!=args.snapshot.name:
            raise ValueError('History cache identity differs')
        return rec,valid_codes(rec)
    old=row.get('existing_history_key')
    if old:
        condition='default' if index==0 else 'anger'
        source=args.existing/'tokens'/(old+'_'+condition+'.json')
        rec=json.loads(source.read_text())
        gen=next(g for g in read_jsonl(args.existing/'generations.jsonl') if g['scenario_id']==old and g['condition']==condition)
        if gen['text_SHA256']!=row['text_sha256'] or gen['model_revision']!=args.snapshot.name or gen['voice']!=protocol['voice']:
            raise ValueError('Historical continuation is not this frozen input/configuration')
        expected=list(model._build_prompt(model.tok,row['text'],protocol['voice'],style='' if index==0 else 'ANGER'))
        if rec['prompt_ids']!=expected:raise ValueError('Historical prompt differs')
        rec.update(source='reused',source_token_file=str(source),source_sha256=digest(source.read_bytes()),
            source_generation=gen,seed=gen['seed'])
    else:
        seed=protocol['history_seed'] if index==0 else protocol['secondary_history_seed']
        prompt=list(model._build_prompt(model.tok,row['text'],protocol['voice'],style=''))
        ids=torch.tensor([prompt],device=model.device);torch.manual_seed(seed)
        cap=token_budget(row['text']);start=time.time()
        with torch.inference_mode():
            result=model.lm.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=cap,
                eos_token_id=model.eos_ids,pad_token_id=model.tok.eos_token_id,do_sample=True,**protocol['decoding'])
        generated=result[0,len(prompt):].tolist()
        rec={'prompt_ids':prompt,'generated_token_ids':generated,'seed':seed,'max_new_tokens':cap,
            'hit_token_cap':len(generated)>=cap,'source':'generated_missing_reference_codes_only','runtime_s':time.time()-start,
            'audio_generated':False,'decoder_executed':False}
    rec.update(text_sha256=row['text_sha256'],model_revision=args.snapshot.name,history_index=index)
    write_json(target,rec)
    return rec,valid_codes(rec)


def inspect_history(model,row,index,record,codes,n,args,protocol,historical,basis):
    dest=args.output/row['id']/f'h{index}'
    if (dest/'completion.json').exists():
        done=json.loads((dest/'completion.json').read_text())
        if done['protocol_sha256']!=digest(args.protocol.read_bytes()):raise ValueError('Completed run protocol differs')
        if any(digest((dest/name).read_bytes())!=sha for name,sha in done['artifacts'].items()):raise ValueError('Completed artifact changed')
        return done
    dest.mkdir(parents=True,exist_ok=False)
    targets=codes[:n];traces={};prob={};phase=np.arange(n)%7;allrows=np.arange(n);genrows=allrows>0
    selected=np.unique(np.r_[np.arange(7),np.arange(n-7,n)]) if row['capture_exact'] or args.stage=='engineering' else np.array([],dtype=int)
    checks={};start=time.time()
    for condition in STYLES:
        prompt=list(model._build_prompt(model.tok,row['text'],protocol['voice'],style=protocol['styles'][condition]))
        tr=trace_context(model.lm,prompt,targets,check_no_hook=condition=='default' and args.stage=='engineering')
        traces[condition]=tr
        pr=probability_context(tr['logits'],phase,device=model.device);prob[condition]=pr
        tops=[{'j':j,'ids':pr['top_ids'][j],'probabilities':pr['top_probabilities'][j],
            'tokens':model.tok.convert_ids_to_tokens(pr['top_ids'][j])} for j in range(n)]
        write_json(dest/(condition+'.json'),{'prompt_ids':prompt,'prompt_length':len(prompt),
            'prompt_token_pieces':model.tok.convert_ids_to_tokens(prompt),'prompt_text':model.tok.decode(prompt),
            'history_tokens':targets,'positions':tr['positions'],'top_predictions':tops,
            'checks':tr['checks'],'evidence':'observational_replay'})
        np.savez_compressed(dest/(condition+'.npz'),**{s:tr['states'][s][:,selected] for s in SITES},
            prediction_rows=selected,prompt_embeddings=tr['extras']['embeddings'][:len(prompt)],
            final_norm=tr['extras']['final_norm'][selected])
        checks[condition]=tr['checks']
        print(json.dumps({'id':row['id'],'history':index,'condition':condition,'positions':n}),flush=True)
    if row['numerical_repeat'] or args.stage=='engineering':
        prompt=json.loads((dest/'default.json').read_text())['prompt_ids']
        repeat=trace_context(model.lm,prompt,targets)
        if not np.array_equal(repeat['logits'],traces['default']['logits']) or any(not np.array_equal(repeat['states'][s],traces['default']['states'][s]) for s in SITES):
            raise ValueError('Identical-context replay is not bit-exact')
        pp=probability_context(repeat['logits'],phase,device=model.device)
        control=probability_contrast(prob['default'],pp,targets,phase)
        maximum=max(r['jsd_nats'] for r in control)
        if maximum>2e-6:raise ValueError('Identical-context JSD exceeds numerical tolerance')
        write_json(dest/'identical_control.json',{'states_exact':True,'logits_exact':True,'max_JSD_nats':maximum,'n_positions':n})
        del repeat,pp
    # Equal-format gate is verified for every input, not assumed from one example.
    lengths={c:len(json.loads((dest/(c+'.json')).read_text())['prompt_ids']) for c in STYLES}
    if len({lengths[c] for c in STYLES if c!='default'})!=1 or lengths['news']-lengths['default']!=4:
        raise ValueError('Styles are not equal-format / Default offset differs')
    means={c:np.stack([phase_pool(traces[c]['states'][s],phase) for s in SITES]) for c in STYLES}
    deltas=np.stack([means[c]-means['news'] for c in ('anger','happy','sad')])
    default_delta=means['default']-means['news']
    if row['study_split']=='development' or (row['second_history'] and index==0):
        np.savez_compressed(dest/'phase_contrasts.npz',news_relative=deltas,default_minus_news=default_delta)
    geometry={'historical_news':np.stack([cosine(deltas[k,0],(historical[c]-historical['control_news'])[:,None,:]) for k,c in enumerate(('anger','happy','sad'))]),
              'historical_default':np.stack([cosine(deltas[k,0]-default_delta[0],historical[c][:,None,:]) for k,c in enumerate(('anger','happy','sad'))])}
    # Full-position historical alignment is cheap to retain, unlike full vectors.
    # The phase-mean-vector cosine above is a different statistic and is labeled
    # separately from these pointwise cosines.
    positional={}
    for reference in ('news','default'):
        positional[reference]=np.stack([cosine(traces[c]['states']['resid']-traces[reference]['states']['resid'],
            (historical[c]-(historical['control_news'] if reference=='news' else 0))[:,None,:])
            for c in ('anger','happy','sad')]).astype(np.float32)
    np.savez_compressed(dest/'historical_alignment_positions.npz',**positional)
    if basis is not None:geometry.update(project_structure(deltas,basis))
    np.savez_compressed(dest/'geometry_metrics.npz',**geometry)
    layer_rows=[];prob_rows=[];decomposition=[];summary=[]
    for reference,condition in CONTRASTS:
        prefix=condition+'_minus_'+reference
        prediction=probability_contrast(prob[reference],prob[condition],targets,phase)
        savecsv(dest/(prefix+'_predictions.csv'),prediction)
        maps={}
        for site in SITES:
            ca,cb=traces[reference]['states'][site],traces[condition]['states'][site]
            cos,l2=state_differences(ca,cb)
            maps[site+'_cosine_distance']=cos.astype(np.float32);maps[site+'_normalized_l2']=l2.astype(np.float32)
            for layer in range(len(ca)):
                for p in range(7):
                    mask=(phase==p)&genrows
                    layer_rows.append({'reference':reference,'condition':condition,'site':site,'layer':layer,'phase':p,
                        'n_positions':int(mask.sum()),'normalized_l2':float(np.mean(l2[layer,mask])),
                        'cosine_distance':float(np.mean(cos[layer,mask]))})
        np.savez_compressed(dest/(prefix+'_heatmaps.npz'),**maps)
        for p in range(7):
            selected_rows=[r for r in prediction if r['j']>0 and r['predicted_phase']==p]
            prob_rows.append({'reference':reference,'condition':condition,'phase':p,'n_positions':len(selected_rows),
                **{k:float(np.mean([r[k] for r in selected_rows])) for k in ('jsd_nats','phase_jsd_nats','delta_target_logp','argmax_changed','reference_entropy_nats','changed_entropy_nats')}})
        if reference=='news':
            # Remove row 0 before decomposition means; all seven phases remain.
            from src.mechanism_study import residual_decomposition
            decomposition.extend({'condition':condition,**r} for r in residual_decomposition(traces[reference],traces[condition],phase))
        summary.append({'reference':reference,'condition':condition,'positions':n,
            'generation_JSD_nats':float(np.mean([r['jsd_nats'] for r in prediction if r['j']>0])),
            'all_position_JSD_nats':float(np.mean([r['jsd_nats'] for r in prediction])),
            'argmax_changed_fraction':float(np.mean([r['argmax_changed'] for r in prediction if r['j']>0]))})
    savecsv(dest/'layer_phase_metrics.csv',layer_rows);savecsv(dest/'probability_phase_metrics.csv',prob_rows)
    savecsv(dest/'residual_decomposition.csv',decomposition);savecsv(dest/'comparison_summary.csv',summary)
    # Compare with the two prior complete traces only when the position grid is
    # exactly the same. Raw distributions had not been saved, so replay is needed.
    prior={'c03':'20261115_v1','c04':'20261115_c04_v1'}.get(row['id'])
    if prior and args.prior_traces:
        old=args.prior_traces/prior/'trace'
        name='prompt_on_default' if index==0 else 'prompt_on_anger'
        original=old/(name+'_predictions.csv')
        with original.open() as f:old_rows=list(csv.DictReader(f))
        if len(old_rows)==n:
            new=next(r for r in summary if r['reference']=='default' and r['condition']=='anger')
            difference=abs(new['all_position_JSD_nats']-np.mean([float(r['jsd_nats']) for r in old_rows]))
            if difference>2e-6:raise ValueError('Old c03/c04 JSD compatibility failed')
            write_json(dest/'prior_compatibility.json',{'source':str(original),'max_mean_JSD_difference':difference,'tolerance':2e-6})
    write_json(dest/'metadata.json',{'id':row['id'],'study_split':row['study_split'],'category':row['category'],
        'history_index':index,'history_provenance':record,'n_positions':n,'generation_rows':'j>=1','phase_weighting':'equal',
        'protocol_sha256':digest(args.protocol.read_bytes()),'text_sha256':row['text_sha256'],'checkpoint':args.snapshot.name,
        'codec':args.codec_snapshot.name,'voice':protocol['voice'],'decoder_replayed':False,
        'style_prompt_lengths':lengths,'checks':checks,'tensor_rows':selected.tolist(),
        'basis_sha256':digest(args.basis.read_bytes()) if args.basis else None})
    files={f.name:digest(f.read_bytes()) for f in sorted(dest.iterdir()) if f.is_file()}
    completion={'completed':True,'id':row['id'],'history':index,'protocol_sha256':digest(args.protocol.read_bytes()),
        'artifacts':files,'runtime_s':time.time()-start,'positions':n,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    write_json(dest/'completion.json',completion)
    print(json.dumps({'completed':row['id'],'history':index,'seconds':completion['runtime_s']}),flush=True)
    return completion,deltas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','protocol','config','snapshot','codec-snapshot','historical','existing','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['engineering','development','confirmation'],required=True)
    p.add_argument('--ids',nargs='+');p.add_argument('--basis',type=Path)
    p.add_argument('--prior-traces',type=Path,help='Optional November 15 trace root for the c03/c04 compatibility check')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65)
    protocol=json.loads(a.protocol.read_text());rows=read_jsonl(a.cohort)
    rows=[r for r in rows if (a.stage=='engineering' and r['id'] in ('c03','c04')) or r['study_split']==a.stage]
    if a.ids:rows=[r for r in rows if r['id'] in a.ids]
    if a.stage=='confirmation' and not a.basis:raise ValueError('Confirmation requires a frozen development basis')
    if a.stage=='confirmation':
        gate=json.loads(a.basis.with_name('confirmation_gate.json').read_text())
        if gate['basis_sha256']!=digest(a.basis.read_bytes()) or gate['protocol_sha256']!=digest(a.protocol.read_bytes()) or gate['cohort_sha256']!=digest(a.cohort.read_bytes()):
            raise ValueError('Frozen confirmation gate identity differs')
    if not rows:raise ValueError('No requested rows')
    _,model=load_model(a)
    with np.load(a.historical,allow_pickle=False) as z:historical={k:v.copy() for k,v in z.items()}
    basis=None
    if a.basis:
        with np.load(a.basis,allow_pickle=False) as z:basis={k:v.copy() for k,v in z.items()}
    a.output.mkdir(parents=True,exist_ok=True)
    for row in rows:
        subject=a.output/row['id'];subject.mkdir(exist_ok=True)
        try:
            histories=[prepare_history(model,row,k,a,protocol,subject) for k in range(2 if row['second_history'] else 1)]
            n=min(protocol['max_frames']*7,*(len(ids) for _,ids in histories))
            if n<protocol['min_frames']*7:raise ValueError('Insufficient valid history frames')
            if len(histories)==2:
                first=np.asarray(histories[0][1][:n]);second=np.asarray(histories[1][1][:n])
                write_json(subject/'history_comparison.json',{'inspected_tokens':n,'identical_inspected_prefix':bool(np.array_equal(first,second)),
                    'token_agreement_fraction':float(np.mean(first==second)),
                    'interpretation':'Two histories of the same text are paired robustness conditions, not independent utterance samples.'})
            primary_delta=None
            for k,(record,codes) in enumerate(histories):
                returned=inspect_history(model,row,k,record,codes,n,a,protocol,historical,basis)
                if isinstance(returned,tuple):done,delta=returned
                else:
                    done=returned
                    f=subject/f'h{k}/phase_contrasts.npz'
                    delta=np.load(f,allow_pickle=False)['news_relative'] if f.exists() else None
                if k==0:primary_delta=delta
                elif delta is not None and primary_delta is not None:
                    np.savez_compressed(subject/'history_robustness.npz',cosine=cosine(primary_delta,delta).astype(np.float32))
            write_json(subject/'subject_complete.json',{'id':row['id'],'histories':len(histories),'status':'complete'})
            if (subject/'failure.json').exists():
                previous=subject/'failure.json';archive=subject/'resolved_failures';archive.mkdir(exist_ok=True)
                previous.rename(archive/(digest(previous.read_bytes())[:16]+'.json'))
        except Exception as e:
            write_json(subject/'failure.json',{'id':row['id'],'type':type(e).__name__,'message':str(e)})
            # A technical blocker halts this bounded chunk; do not silently run
            # the remaining subjects or change generation/extraction settings.
            raise
    write_json(a.output/(a.stage+'_chunk_'+(a.ids[0] if a.ids else 'all')+'.json'),{'completed_ids':[r['id'] for r in rows],
        'protocol_sha256':digest(a.protocol.read_bytes()),'basis_sha256':digest(a.basis.read_bytes()) if a.basis else None})


if __name__=='__main__':main()
