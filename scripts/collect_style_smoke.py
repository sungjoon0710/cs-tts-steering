"""Collect one construction pair only; no vector fitting or generation."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,read_jsonl,write_json
from src.representation_probe import collect_phases
from src.tts import IndicSpeak


def main(a):
    if a.output.exists():raise ValueError('Preserve previous smoke outputs; choose a new directory')
    a.output.mkdir(parents=True)
    cfg=json.loads((a.source/'config.json').read_text())
    if digest((a.source/'inputs.csv').read_bytes())!=cfg['manifest_SHA256']:raise ValueError('Input manifest changed')
    records={r['generation_id']:r for r in read_jsonl(a.source/'generations.jsonl')}
    pair=[r for r in records.values() if (r['utterance_id'],r['voice'],r['script'])==(cfg['smoke_input'],'Amit','devanagari')]
    if len(pair)!=2 or {r['style'] for r in pair}!={'','ANGER'} or any(r['split']!='construct' or not r['technical_pass'] for r in pair):raise ValueError('Use exactly one valid construction pair')
    saved={}
    for r in pair:
        p=a.source/r['tokens_path']
        if digest(p.read_bytes())!=r['tokens_SHA256']:raise ValueError('Saved token file changed')
        saved[r['style']]=json.loads(p.read_text())
    snapshot=a.snapshot
    if snapshot.name!=cfg['model'][1] or digest((snapshot/'inference.py').read_bytes())!=cfg['vendor_inference_SHA256']:raise ValueError('Checkpoint/vendor mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA smoke test requires an available GPU')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(.9,18*1024**3/torch.cuda.get_device_properties(0).total_memory))
    runner=IndicSpeak(device='cuda',snapshot=snapshot);lm=runner.tts.lm
    neutral=next(r for r in pair if r['style']=='')
    prompt=runner._build_prompt(runner.tts.tok,neutral['input_text'],neutral['voice'],style='')
    if prompt!=saved['']['prompt_token_ids']:raise ValueError('Neutral prompt differs from saved trajectory')
    start=time.monotonic();results=[];baseline=None
    for style in ['','ANGER']:
        torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        arrays,meta=collect_phases(lm,prompt,saved[style]['generated_token_ids'],chunk_size=256)
        name='ANGER' if style else 'empty';target=a.output/(name+'_phase_statistics.npz');np.savez_compressed(target,**arrays)
        r=next(r for r in pair if r['style']==style)
        meta.update(generation_id=r['generation_id'],utterance_id=r['utterance_id'],split='construct',voice='Amit',script='devanagari',
            source_token_file_SHA256=r['tokens_SHA256'],neutral_prompt_SHA256=digest(json.dumps(prompt).encode()),
            source_recording_style=style,model_revision=cfg['model'][1],saved_statistics_SHA256=digest(target.read_bytes()),
            runtime_s=time.monotonic()-t,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            seed=cfg['seed'],TTS_decoding=cfg['decoding'])
        write_json(target.with_suffix('.json'),meta);results.append(meta)
        if not style:baseline=arrays
        print(json.dumps({'collected':name,'frames_per_phase':meta['phase_counts'][0][0],'runtime_s':meta['runtime_s']}),flush=True)
    check,_=collect_phases(lm,prompt,saved['']['generated_token_ids'],chunk_size=128)
    relative=float(np.linalg.norm(check['means']-baseline['means'])/max(np.linalg.norm(baseline['means']),1e-12))
    same_counts=bool(np.array_equal(check['counts'],baseline['counts']))
    passed=same_counts and relative<=.02
    write_json(a.output/'smoke_result.json',{'passed':passed,'records_collected':2,'construction_sentences':1,'heldout_sentences_touched':0,
        'same_empty_prompt_both_continuations':len({r['neutral_prompt_SHA256'] for r in results})==1,
        'chunk_128_vs_256_mean_relative_L2':relative,'predeclared_relative_L2_tolerance':.02,'chunk_counts_exact':same_counts,
        'GPU':torch.cuda.get_device_name(0),'runtime_s':time.monotonic()-start,'vectors_fitted':False,
        'nonzero_steering_run':False,'new_recordings_generated':False,'style_contrast_human_validated':False,
        'source_SHA256':{'runner':digest(Path(__file__).read_bytes()),'collector':digest((Path(__file__).parents[1]/'src/representation_probe.py').read_bytes())}})
    if not passed:raise RuntimeError('Replay chunk-size smoke check failed; inspect statistics before collecting more')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','output','snapshot']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
