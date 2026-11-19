"""Frozen six-text extension and two-layer prompt-anchor causal feasibility test."""
import argparse,json,shutil,sys,time,warnings
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tts import IndicSpeak
from src.evaluation.common import read_jsonl,write_json,digest
from scripts.steering_pilot import structure,prompt_checks,VENDOR_SHA
from src.position_steering import AnchorIntervention

def main(a):
 root=a.root.resolve();study=a.output.resolve();cfg=json.loads((root/'config.json').read_text());sentences=read_jsonl(root/'scenarios.jsonl')
 decision=json.loads(a.design.read_text());run=study/decision['directories'][a.stage]
 run.mkdir(parents=True,exist_ok=False);(run/'audio').mkdir();(run/'tokens').mkdir()
 write_json(run/'config.json',cfg);shutil.copyfile(root/'scenarios.jsonl',run/'scenarios.jsonl')
 snap=a.snapshot
 if snap.name!=cfg['model'][1] or digest((snap/'inference.py').read_bytes())!=VENDOR_SHA:
  raise ValueError('Use the frozen Indic-Speak snapshot and inference source')
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(.9,16*1024**3/torch.cuda.get_device_properties(0).total_memory))
 runner=IndicSpeak(device='cuda',snapshot=snap);lm=runner.tts.lm;lm.eval();assert runner.decoding==cfg['decoding']
 records=[]
 def save(r):
  records.append(r)
  with (run/'generations.jsonl').open('a') as fp:fp.write(json.dumps(r)+'\n')
 def generate(s,condition,style='',intervention=None):
  gid=s['id']+'_'+condition;prompt=list(runner._build_prompt(runner.tts.tok,s['reply'],'Amit',style=style));captured={}
  def capture(pids,tokens):
   assert list(pids)==prompt;captured.update(prompt_ids=list(pids),generated_token_ids=list(tokens),token_structure=structure(tokens,{runner.tts._eos,runner.tts.tok.eos_token_id},s['max_new_tokens']))
   write_json(run/'tokens'/(gid+'.json'),captured)
  t=time.monotonic();r={'generation_id':gid,'scenario_id':s['id'],'split':s['split'],'condition':condition,'style':style,'voice':'Amit','input_text':s['reply'],
   'text_SHA256':s['text_SHA256'],'seed':cfg['seed'],'max_new_tokens':s['max_new_tokens'],'model_revision':cfg['model'][1],'status':'failed'}
  try:
   assert all(prompt_checks(runner.tts.tok,s['reply'],'Amit',style,prompt,runner._build_prompt).values())
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    if intervention:
     layer,direction,epsilon=intervention
     with AnchorIntervention(lm,prompt,layer,torch.as_tensor(direction),epsilon) as h:
      wav,info=runner(s['reply'],'Amit',cfg['seed'],s['max_new_tokens'],style=style,token_callback=capture)
     r['intervention']=h.result();assert h.selected==1
    else:wav,info=runner(s['reply'],'Amit',cfg['seed'],s['max_new_tokens'],style=style,token_callback=capture)
   assert wav.ndim==1 and np.isfinite(wav).all() and np.any(wav);path=run/'audio'/(gid+'.wav');sf.write(path,wav,24000,subtype='PCM_16')
   st=captured['token_structure'];duration=len(wav)/24000
   passed=bool(1<=duration<=25 and st['normal_EOS'] and not st['token_cap_reached'] and st['invalid_frames']==st['incomplete_trailing_tokens']==0)
   r.update(info,status='generated',technical_pass=passed,duration_s=duration,audio_path=path.relative_to(run).as_posix(),audio_SHA256=digest(path.read_bytes()),
      token_structure=st,warnings=[str(w.message) for w in caught])
  except Exception as e:r.update(technical_pass=False,error_type=type(e).__name__,error=str(e)[:600])
  r['runtime_s']=time.monotonic()-t;save(r);print(json.dumps({'stage':a.stage,'id':gid,'technical_pass':r['technical_pass']}),flush=True)
  if not r['technical_pass']:raise RuntimeError('Technical degradation; stop without changing strength')
  return r,captured
 if a.stage=='expand':
  prior={r['generation_id']:r for r in read_jsonl(root/'generations.jsonl')}
  for s in sentences:
   for c,style in cfg['styles'].items():
    gid=s['id']+'_'+c
    if gid in prior:
     r=prior[gid];assert r['input_text']==s['reply'] and r['seed']==cfg['seed'] and r['model_revision']==cfg['model'][1]
     shutil.copyfile(root/r['audio_path'],run/r['audio_path']);shutil.copyfile(root/'tokens'/(gid+'.json'),run/'tokens'/(gid+'.json'))
     assert digest((run/r['audio_path']).read_bytes())==r['audio_SHA256'];save({**r,'reused_from_screen':True})
    else:generate(s,c,style)
 else:
  construction=[s for s in sentences if s['split']=='calibration'];assert [s['id'] for s in construction]==['c01','c02','c03','c04']
  vector_dir=study/'causal_vectors'
  if a.stage=='smoke':
   vector_dir.mkdir();raw={};random={}
   for layer in decision['layers']:
    differences=[];contributions=[]
    for s in construction:
     with np.load(root/'activations'/(s['id']+'_anger.npz')) as x,np.load(root/'activations'/(s['id']+'_default.npz')) as y:
      d=x['residual'][layer].astype(np.float64)-y['residual'][layer].astype(np.float64)
     differences.append(d);contributions.append({'id':s['id'],'delta_norm':float(np.linalg.norm(d))})
    v=np.mean(differences,axis=0).astype(np.float32);raw[str(layer)]=v
    rng=np.random.default_rng(decision['random_seed']+layer);w=rng.normal(size=3072);w=w/np.linalg.norm(w)*np.linalg.norm(v);random[str(layer)]=w.astype(np.float32)
    write_json(vector_dir/f'layer_{layer}.json',{'layer':layer,'direction_norm':float(np.linalg.norm(v)),'construction':contributions,'heldout_used':False,
      'site':'post-block final speech-start PROMPT state','interpretation':'ANGER prompt context contrast; not a validated acoustic anger vector','norm_fraction':decision['epsilon'],
      'all_four_calibration_prompts_used':True,'random_seed':decision['random_seed']+layer})
   np.savez_compressed(vector_dir/'prompt_contrast.npz',**raw);np.savez_compressed(vector_dir/'random_matched_norm.npz',**random)
  raw=dict(np.load(vector_dir/'prompt_contrast.npz',allow_pickle=False));random=dict(np.load(vector_dir/'random_matched_norm.npz',allow_pickle=False))
  held=[s for s in sentences if s['split']=='heldout'];cases=[(s,layer,kind) for s in held for layer in decision['layers'] for kind in ['positive','negative','random']]
  if a.stage=='smoke':
   s=held[0];layer=decision['layers'][0];reference=next(x for x in read_jsonl(study/'expanded/generations.jsonl') if x['generation_id']==s['id']+'_default')
   zero,tokens=generate(s,f'L{layer:02d}_zero_check',intervention=(layer,raw[str(layer)],0))
   baseline=json.loads((study/'expanded/tokens'/(s['id']+'_default.json')).read_text())
   assert tokens['generated_token_ids']==baseline['generated_token_ids'] and zero['audio_SHA256']==reference['audio_SHA256']
   write_json(run/'zero_equivalence.json',{'generated_tokens_exact':True,'PCM_WAV_hash_exact':True,'fresh_cache':True,'hooks_removed':True})
   cases=cases[:1]
  else:cases=cases[1:]
  for s,layer,kind in cases:
   epsilon=decision['epsilon']*(-1 if kind=='negative' else 1);direction=random[str(layer)] if kind=='random' else raw[str(layer)]
   generate(s,f'L{layer:02d}_{kind}',intervention=(layer,direction,epsilon))
 write_json(run/'completion.json',{'stage':a.stage,'records':len(records),'technical_pass':sum(r['technical_pass'] for r in records),
     'new_generation_calls':sum(not r.get('reused_from_screen',False) for r in records),'emotion_perceptually_validated':False})

if __name__=='__main__':
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root', type=Path, required=True, help='Private saved screen run')
 parser.add_argument('--output', type=Path, required=True, help='Separate output root for the three stages')
 parser.add_argument('--snapshot', type=Path, required=True)
 parser.add_argument('--design', type=Path, default=Path(__file__).resolve().parents[1]/'configs/research/nov12/anchor_intervention.json')
 parser.add_argument('--stage', choices=['expand','smoke','rest'], required=True)
 args = parser.parse_args()
 if args.root.resolve() == args.output.resolve() or args.output.resolve().is_relative_to(args.root.resolve()):
  parser.error('--output must be outside the historical input directory')
 main(args)
