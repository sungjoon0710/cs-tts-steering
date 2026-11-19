"""Bounded causal replay on saved valid histories; no speech decoder execution."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.causal_patch import Patch,replay,score_patch,jsd,code_logp
from src.evaluation.common import digest,read_jsonl,write_json
from scripts.inspect_representations import load_model
from scripts.run_m1 import valid_codes
from src.tts_settings import token_budget

STYLES=['news','anger','happy','sad'];LAYERS=[5,11,26,27]
SITES=[(l,s) for l in LAYERS for s in ['attn','mlp','resid']]+[(l,'resid') for l in [4,10,25]]


def case_list(plan):
    cases=[]
    for rule in plan['rules']:
        for donor in rule.get('donors',STYLES[1:]):
            for frame in rule.get('frames',[4,18]):
                for phase in rule.get('phases',[0]):
                    for variant in rule['variants']:
                        reverse=variant=='reverse';kind='native' if reverse else variant
                        spec=dict(layer=rule['layer'],site=rule['site'],j=frame*7+phase,frame=frame,phase=phase,
                                  recipient=donor if reverse else 'news',donor='news' if reverse else donor,variant=kind,
                                  comparison='reverse' if reverse else kind,style=donor)
                        spec['case_id']=digest(json.dumps(spec,sort_keys=True).encode())[:20]
                        cases.append(spec)
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate intervention case')
    if any(c['j']==0 or not 0<=c['j']<168 for c in cases):raise ValueError('M2 cases require generation positions inside the fixed 24-frame window')
    return cases


def history(model,row,h,args,protocol):
    if row['study_split']=='development':
        p=args.m1_root/'study'/row['id']/f'history_{h}.json';record=json.loads(p.read_text())
        if record['text_sha256']!=row['text_sha256'] or record['model_revision']!=args.snapshot.name:raise ValueError('M1 history identity differs')
        record={**record,'m1_source':str(p),'m1_source_sha256':digest(p.read_bytes())}
    else:
        p=args.output/'histories'/row['id']/f'history_{h}.json'
        if p.exists():record=json.loads(p.read_text())
        else:
            p.parent.mkdir(parents=True,exist_ok=True)
            seed=protocol['history_seeds'][h];prompt=list(model._build_prompt(model.tok,row['text'],'Amit',style=''))
            ids=torch.tensor([prompt],device=model.device);torch.manual_seed(seed);started=time.time();cap=token_budget(row['text'])
            with torch.inference_mode():
                out=model.lm.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=cap,eos_token_id=model.eos_ids,
                    pad_token_id=model.tok.eos_token_id,do_sample=True,**protocol['decoding'])
            tokens=out[0,len(prompt):].tolist()
            record=dict(prompt_ids=prompt,generated_token_ids=tokens,seed=seed,max_new_tokens=cap,hit_token_cap=len(tokens)>=cap,
                text_sha256=row['text_sha256'],model_revision=args.snapshot.name,history_index=h,runtime_s=time.time()-started,
                audio_generated=False,decoder_executed=False,source='M2_missing_default_reference_codes_only')
            write_json(p,record)
    codes=valid_codes(record)
    if record['text_sha256']!=row['text_sha256'] or record['model_revision']!=args.snapshot.name:raise ValueError('Cached reference identity differs')
    if len(codes)<168:raise ValueError('Completed reference is shorter than the predeclared 24-frame window')
    return record,codes[:168]


def baseline(model,row,h,record,codes,args,protocol):
    folder=args.output/'baselines'/row['id']/f'h{h}';rows=protocol['prediction_rows']
    key=digest(json.dumps(dict(text=row['text_sha256'],history=record['generated_token_ids'],rows=rows,
        checkpoint=args.snapshot.name,styles=protocol['styles'],capture_sites=SITES),sort_keys=True).encode())
    meta=folder/'metadata.json';data=folder/'baseline.npz'
    prompts={c:list(model._build_prompt(model.tok,row['text'],'Amit',style=protocol['styles'][c])) for c in STYLES}
    if len({len(p) for p in prompts.values()})!=1:raise ValueError('Styled prompt lengths are not matched')
    started=time.time();arrays={};checks={};runtimes={}
    if meta.exists():
        metadata=json.loads(meta.read_text())
        if metadata['context_key']!=key or digest(data.read_bytes())!=metadata['array_sha256']:raise ValueError('Cached causal baseline changed')
        with np.load(data,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z}
    else:
        folder.mkdir(parents=True,exist_ok=False)
        for c in STYLES:
            result=replay(model.lm,prompts[c],codes,rows,SITES)
            arrays[c+'__logits']=result['logits'].cpu().numpy().copy();checks[c]=result['checks'];runtimes[c]=result['checks']['runtime_s']
            for (layer,site),state in result['states'].items():arrays[f'{c}__{layer}__{site}']=state
            # Actual two BF16-add order, using captured previous-block inputs.
            for layer in LAYERS:
                incoming=torch.from_numpy(result['states'][(layer-1,'resid')]).to(dtype=model.lm.dtype)
                attention=torch.from_numpy(result['states'][(layer,'attn')]).to(dtype=model.lm.dtype)
                mlp=torch.from_numpy(result['states'][(layer,'mlp')]).to(dtype=model.lm.dtype)
                if not np.array_equal(((incoming+attention)+mlp).float().numpy(),result['states'][(layer,'resid')]):
                    raise ValueError('M2 baseline BF16 residual reconstruction failed')
        np.savez_compressed(data,**arrays)
        metadata=dict(context_key=key,id=row['id'],study_split=row['study_split'],history=h,text_sha256=row['text_sha256'],
            prompts=prompts,history_provenance=record,inspected_codes=codes,prediction_rows=rows,capture_sites=SITES,
            checkpoint=args.snapshot.name,codec=args.codec_snapshot.name,dtype=str(model.lm.dtype),checks=checks,
            source_script_sha256=digest(Path(__file__).read_bytes()),array_sha256=digest(data.read_bytes()),
            pass_runtimes_s=runtimes,runtime_s=time.time()-started,gpu=torch.cuda.get_device_name())
        write_json(meta,metadata)
    logits={c:torch.from_numpy(arrays[c+'__logits']).to(model.device) for c in STYLES}
    lp={c:x.log_softmax(-1) for c,x in logits.items()}
    states={c:{(l,s):arrays[f'{c}__{l}__{s}'] for l,s in SITES} for c in STYLES}
    return dict(metadata=metadata,prompts=prompts,logits=logits,lp=lp,states=states,rows=rows,folder=folder)


def execute(model,base,codes,spec,protocol,basis,save_vector=False):
    j=spec['j'];k=base['rows'].index(j);site=(spec['layer'],spec['site']);recipient=spec['recipient'];donor=spec['donor']
    before=base['states'][recipient][site][k];after=base['states'][donor][site][k]
    direction=None
    if spec['variant'] in ('shared','specific'):
        si=['resid','attn','mlp'].index(spec['site']);phase=j%7
        direction=basis['shared'][si,spec['layer'],phase] if spec['variant']=='shared' else basis['specific'][STYLES[1:].index(donor),si,spec['layer'],phase]
    seed=int(hashlib.sha256((base['metadata']['id']+str(base['metadata']['history'])+spec['case_id']).encode()).hexdigest()[:8],16)
    patch=Patch(spec['layer'],spec['site'],j,before,after,spec['variant'],spec.get('alpha',1.),direction,seed,base['metadata']['dtype'])
    downstream=[(l,'resid') for l in LAYERS if l>=spec['layer']]
    result=replay(model.lm,base['prompts'][recipient],codes,base['rows'],downstream,patch,[j])
    earlier=[i for i,r in enumerate(base['rows']) if r<j]
    previous_error=float((result['logits'][earlier]-base['logits'][recipient][earlier]).abs().max())
    if previous_error!=0:raise ValueError('A patch changed predictions from earlier causal positions')
    if spec['site']=='resid' and spec['layer']==27 and spec['variant']=='native' and spec.get('alpha',1.)==1:
        error=float((result['logits'][k]-base['logits'][donor][k]).abs().max())
        if error>1e-5:raise ValueError('Layer-27 residual positive control did not reproduce donor logits')
    else:error=None
    changed=result['logits'][k].log_softmax(-1)
    metrics=score_patch({c:lp[k] for c,lp in base['lp'].items()},changed,recipient,donor,j%7,codes[j],protocol.get('normalization_floor',1e-6))
    for label in STYLES:
        p=base['lp'][label][k].exp();offset=128266+(j%7)*4096
        metrics[label+'_phase_mass']=float(p[offset:offset+4096].sum());metrics[label+'_total_stop']=float(p[128258]+p[128001])
    metrics['boundary_flag']=min(metrics['news_phase_mass'],metrics[spec['style']+'_phase_mass'])<.99 or max(metrics['news_total_stop'],metrics[spec['style']+'_total_stop'])>.01
    propagation=[]
    for (layer,_),state in result['states'].items():
        actual=state[0].astype(float);n=base['states'][recipient][(layer,'resid')][k].astype(float);d=base['states'][donor][(layer,'resid')][k].astype(float)
        delta=actual-n;target=d-n;den=np.linalg.norm(delta)*np.linalg.norm(target)
        propagation.append(dict(layer=layer,relative_change=float(np.linalg.norm(delta)/np.linalg.norm(n)),
            target_alignment=float(delta@target/den) if den else None,distance_to_donor=float(np.linalg.norm(actual-d))))
    following={}
    for row in [j+1,167]:
        if row in base['rows'] and row!=j:
            ix=base['rows'].index(row);following[str(row)]=float(jsd(base['lp'][recipient][ix],result['logits'][ix].log_softmax(-1)))
    record={**spec,**metrics,**result['checks'], 'random_seed':seed,'earlier_prediction_max_abs_diff':previous_error,
        'last_residual_donor_logit_max_abs_diff':error,'propagation':propagation,'subsequent_selected_row_JSD':following}
    return record,changed.exp().cpu().numpy().copy()


def engineering(model,base,codes,args,protocol,basis):
    output=args.output/'engineering';output.mkdir(exist_ok=False);records=[]
    plain=replay(model.lm,base['prompts']['news'],codes,base['rows'])
    no_hook_error=float((plain['logits']-base['logits']['news']).abs().max())
    if no_hook_error!=0:raise ValueError('No-hook baseline differs from read-only collection')
    spec=dict(layer=5,site='attn',j=28,frame=4,phase=0,recipient='news',donor='anger',variant='native',comparison='native',style='anger',case_id='engineering_native')
    k=base['rows'].index(28);before=base['states']['news'][(5,'attn')][k]
    identity=replay(model.lm,base['prompts']['news'],codes,base['rows'],patch=Patch(5,'attn',28,before,before,dtype=base['metadata']['dtype']))
    identity_error=float((identity['logits']-base['logits']['news']).abs().max())
    if identity_error!=0:raise ValueError('Identity patch changed outputs')
    probabilities=[]
    for name,changes in [('zero',dict(alpha=0.)),('native',{}),('random',dict(variant='random')),('reverse_add',dict(alpha=-1.)),
                         ('reverse',dict(recipient='anger',donor='news')),('final_residual',dict(layer=27,site='resid'))]:
        record,p=execute(model,base,codes,{**spec,**changes,'case_id':'engineering_'+name},protocol,basis)
        if name=='zero':
            changed=replay(model.lm,base['prompts']['news'],codes,base['rows'],patch=Patch(5,'attn',28,before,base['states']['anger'][(5,'attn')][k],alpha=0,dtype=base['metadata']['dtype']))
            zero_error=float((changed['logits']-base['logits']['news']).abs().max())
            if zero_error!=0:raise ValueError('Zero intervention changed outputs')
        records.append(record);probabilities.append(p)
    np.savez_compressed(output/'intervention_probabilities.npz',probabilities=np.stack(probabilities))
    report=dict(status='passed',id=base['metadata']['id'],history=base['metadata']['history'],no_hook_max_abs_diff=no_hook_error,
        identity_max_abs_diff=identity_error,zero_max_abs_diff=zero_error,positions=base['rows'],window_frames=24,
        checkpoint=args.snapshot.name,model_layers=len(model.layers),hidden_size=model.lm.config.hidden_size,dtype=str(model.lm.dtype),
        gpu=torch.cuda.get_device_name(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),cases=records,mean_patch_seconds=float(np.mean([r['runtime_s'] for r in records])),
        earlier_causal_tolerance=0.,identity_tolerance=0.,last_residual_logit_tolerance=1e-5,normalization_projection_norm_tolerance=.01)
    write_json(output/'engineering.json',report);print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['cohort','protocol','config','snapshot','codec-snapshot','basis','m1-root','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan',type=Path);parser.add_argument('--gate',type=Path);parser.add_argument('--engineering',action='store_true');parser.add_argument('--ids',nargs='+')
    parser.add_argument('--history',type=int,default=0);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    protocol=json.loads(args.protocol.read_text());cohort=read_jsonl(args.cohort)
    if any(r['study_split']=='confirmation' for r in cohort):
        if not args.gate:raise ValueError('Independent confirmation requires the frozen design gate')
        gate=json.loads(args.gate.read_text());role='secondary' if args.history else 'primary'
        required={'cohort_'+role+'_sha256':args.cohort,'protocol_sha256':args.protocol,'plan_'+role+'_sha256':args.plan,
                  'basis_sha256':args.basis,'run_script_sha256':Path(__file__),
                  'patch_engine_sha256':Path(__file__).resolve().parents[1]/'src/causal_patch.py'}
        for key,path in required.items():
            if path is None or digest(path.read_bytes())!=gate[key]:raise ValueError('Frozen confirmation input/source mismatch: '+key)
        if gate.get('fitted_on_confirmation') is not False:raise ValueError('Confirmation design does not establish development-only selection')
    if args.ids:cohort=[r for r in cohort if r['id'] in args.ids]
    if not cohort:raise ValueError('No requested subjects')
    config,model=load_model(args,json.loads(args.config.read_text()))
    with np.load(args.basis,allow_pickle=False) as z:basis={k:z[k].copy() for k in z}
    plan=json.loads(args.plan.read_text()) if args.plan else None
    if not args.engineering and plan is None:raise ValueError('A bounded, saved plan is required')
    args.output.mkdir(parents=True,exist_ok=True)
    for row in cohort:
        try:
            try:record,codes=history(model,row,args.history,args,protocol)
            except ValueError as e:
                if str(e) not in ['Reference continuation lacks normal EOS or has trailing/invalid codes',
                    'Invalid/incomplete/non-contiguous speech-code frames','Completed reference is shorter than the predeclared 24-frame window']:raise
                fail=args.output/'failures'/('reference_'+row['id']+f'_h{args.history}.json');fail.parent.mkdir(exist_ok=True)
                write_json(fail,dict(id=row['id'],history=args.history,status='unusable_reference',message=str(e),replaced=False))
                print(json.dumps(dict(reference_failure=row['id'],history=args.history,message=str(e))),flush=True);continue
            base=baseline(model,row,args.history,record,codes,args,protocol)
            if args.engineering:engineering(model,base,codes,args,protocol,basis);return
            folder=args.output/'runs'/plan['name']/row['id']/f'h{args.history}';done=folder/'completion.json'
            if done.exists():
                metadata=json.loads(done.read_text())
                if metadata['plan_sha256']!=digest(args.plan.read_bytes()) or any(digest((folder/n).read_bytes())!=v for n,v in metadata['artifacts'].items()):raise ValueError('Completed M2 case artifacts changed')
                continue
            if folder.exists() and any(folder.iterdir()):raise ValueError('Incomplete case folder contains artifacts; preserve for inspection')
            folder.mkdir(parents=True,exist_ok=True);records=[];probs=[];exclusions=[];started=time.time()
            for spec in case_list(plan):
                try:result,p=execute(model,base,codes,spec,protocol,basis)
                except ValueError as e:
                    # A frozen template may have no BF16-realizable 1%-norm match.
                    # Preserve this off-manifold exclusion; never relax primary random controls.
                    if spec['variant'] not in ('shared','specific') or str(e)!='Cannot match realized perturbation norm within 1%':raise
                    exclusions.append(dict(spec,id=row['id'],history=args.history,status='not_executed_BF16_norm_ineligible',reason=str(e)))
                    continue
                result.update(id=row['id'],study_split=row['study_split'],category=row['category'],history=args.history)
                records.append(result);probs.append(p)
            write_json(folder/'exclusions.json',exclusions)
            write_json(folder/'results.json',records);np.savez_compressed(folder/'probabilities.npz',probabilities=np.stack(probs))
            write_json(done,dict(id=row['id'],history=args.history,plan_sha256=digest(args.plan.read_bytes()),protocol_sha256=digest(args.protocol.read_bytes()),
                baseline_context_key=base['metadata']['context_key'],cases=len(records),planned_cases=len(case_list(plan)),excluded_cases=len(exclusions),runtime_s=time.time()-started,
                artifacts={p.name:digest(p.read_bytes()) for p in folder.iterdir() if p.is_file()},source_sha256=digest(Path(__file__).read_bytes())))
            print(json.dumps(dict(completed=row['id'],history=args.history,plan=plan['name'],cases=len(records),seconds=time.time()-started)),flush=True)
        except Exception as e:
            fail=args.output/'failures'/((plan['name'] if plan else 'engineering')+'_'+row['id']+f'_h{args.history}.json')
            fail.parent.mkdir(exist_ok=True);write_json(fail,dict(id=row['id'],history=args.history,error_type=type(e).__name__,message=str(e)))
            raise


if __name__=='__main__':main()
