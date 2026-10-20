"""Bounded, pinned TTSDS2 attempt on existing audio in a separate environment.

Run only after approving checkpoints and shared-GPU use. Does not install packages,
change official scoring, overwrite results, or write beside collaborator audio.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback,urllib.request,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,private_directory,read_jsonl,write_json

PINS={
    'utter-project/mHuBERT-147':('7ad3fc0bc5106c58c9c13526abccad527150d135','model.safetensors'),
    'facebook/wav2vec2-large-xlsr-53':('c3f9d884181a224a6ac87bf8885c84d1cff3384f','pytorch_model.bin'),
    'facebook/wav2vec2-xlsr-53-espeak-cv-ft':('2c733782da5604684829819a5eb744c193fe9398','pytorch_model.bin'),
    'microsoft/wavlm-base-plus':('4c66d4806a428f2e922ccfa1a962776e232d487b','pytorch_model.bin'),
    'cdminix/masked_prosody_model':('b8230cf6faac6ac3495bcb9ebad15fdebf13ad38','pytorch_model.bin'),
    'pyannote/wespeaker-voxceleb-resnet34-LM':('837717ddb9ff5507820346191109dc79c958d614','pytorch_model.bin'),
    'openai/whisper-small':('973afd24965f72e36ca33b3055d56a652f456b4d','model.safetensors'),
}
NOISE={
    'noise_all_ones.tar.gz':'2f9b277399bc0874aa424b29bad8a5000ea643315e34294d9e68c0534bb33a4f',
    'noise_all_zeros.tar.gz':'df4a21b2589f733ffa2a8974f8427881bf5634319586498faaafd21be359e487',
    'noise_normal_distribution.tar.gz':'59eda7a931d0dfbdcca8373d04d8e7a8379541694fba9a2f9bff2db8a77d77b8',
    'noise_uniform_distribution.tar.gz':'d9f4c064d4963955591d319517b5c40a2f009215c7e98a0ba2cdf238fcf5292e',
}

def prefetch(cache,out,cap):
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi(token=False);plan=[];additional=0
    pins=dict(PINS)
    # Feature-extractor configuration only: never fetch this large HuBERT model.
    name='facebook/hubert-large-ls960-ft'
    pins[name]=(api.model_info(name,token=False).sha,None)
    for name,(revision,weight) in pins.items():
        info=api.model_info(name,revision=revision,files_metadata=True,token=False)
        files=[s for s in info.siblings if s.rfilename==weight or (s.size is not None and s.size<5_000_000 and s.rfilename.endswith(('.json','.txt','.yaml','.yml')))]
        if weight and not any(s.rfilename==weight for s in files):raise ValueError('Pinned weight missing: '+name)
        folder=cache/('models--'+name.replace('/','--'));ref=folder/'refs/main'
        if ref.exists() and ref.read_text().strip()!=revision:raise ValueError('Conflicting cached main revision: '+name)
        for f in files:
            cached=folder/'snapshots'/revision/f.rfilename
            if not cached.exists():additional+=f.size
        plan.append({'model':name,'revision':revision,'files':{s.rfilename:s.size for s in files}})
    # Reserve room for the small phone model, noise references and dependencies.
    if additional+350_000_000>cap:raise ValueError('Additional-download cap would be exceeded')
    write_json(out/'download_plan.json',{'cap_bytes':cap,'additional_HF_bytes':additional,'reserved_other_bytes':350_000_000,'models':plan})
    receipts=[]
    for p in plan:
        snapshot=Path(snapshot_download(p['model'],revision=p['revision'],cache_dir=str(cache),allow_patterns=list(p['files']),token=False,max_workers=2))
        ref=snapshot.parents[1]/'refs/main';ref.parent.mkdir(exist_ok=True)
        if not ref.exists():ref.write_text(p['revision'])
        receipts.extend({'model':p['model'],'revision':p['revision'],'file':f,'bytes':(snapshot/f).stat().st_size,'sha256':digest((snapshot/f).read_bytes())} for f in p['files'])
        print('Pinned cache ready:',p['model'],flush=True)
    # Download/extract the official phone archive ourselves with size/path checks.
    import allosaurus
    target=Path(allosaurus.__file__).parent/'pretrained'
    if list(target.glob('*')):raise ValueError('Unexpected existing Allosaurus model; verify version explicitly')
    archive=out/'allosaurus_latest.tar.gz';limit=50_000_000
    url='https://github.com/xinjli/allosaurus/releases/download/v1.0/latest.tar.gz'
    h=hashlib.sha256();count=0
    with urllib.request.urlopen(url,timeout=60) as response,archive.open('xb') as f:
        while chunk:=response.read(1024*1024):
            count+=len(chunk)
            if count>limit:raise ValueError('Phone-model download exceeds reservation')
            h.update(chunk);f.write(chunk)
    if h.hexdigest()!='7ccf374aba1dde19b527a988709c59ed9697ab1c1df231f2b2428dd81784a316':raise ValueError('Pinned phone-model archive changed')
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path=target/member.name
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()) or target.resolve() not in path.resolve().parents:raise ValueError('Unsafe phone-model archive')
        tar.extractall(target)
    write_json(out/'checkpoint_manifest.json',{'files':receipts,'allosaurus':{'url':url,'bytes':count,'sha256':h.hexdigest()},'note':'Only owned cache refs were created; actual execution is offline.'})

def evaluate(a,out):
    import numpy as np,torch
    from ttsds import BenchmarkSuite
    from ttsds.util.dataset import DirectoryDataset
    from ttsds.util.distances import frechet_distance,wasserstein_distance
    # Exercise the actual official distance path, not its unused fast alternative.
    assert abs(frechet_distance((np.zeros(2),np.eye(2)),(np.ones(2),np.eye(2)))-2)<1e-5
    assert abs(wasserstein_distance(np.array([0.,1.]),np.array([1.,2.]))-1)<1e-8
    rows=read_jsonl(a.baseline_run/'tables/metrics.jsonl')
    if len(rows)!=300 or len({r['generation_id'] for r in rows})!=300:raise ValueError('Expected 300 unique baseline generations')
    matched={r['utterance_id'] for r in rows}
    if len(matched)!=100:raise ValueError('Expected 100 matched input IDs')
    datasets=[]
    for config in sorted({r['configuration_id'] for r in rows}):
        rs=[r for r in rows if r['configuration_id']==config]
        if len(rs)!=100 or {r['utterance_id'] for r in rs}!=matched:raise ValueError('Expected 100 matched inputs per configuration')
        for r in rs:
            if digest(Path(r['source_audio_path']).read_bytes())!=r['audio']['audio_sha256']:raise ValueError('Changed baseline audio')
        parents={Path(r['source_audio_path']).parent for r in rs}
        if len(parents)!=1:raise ValueError('Expected one exact baseline directory per configuration')
        # Official WavListDataset embeds absolute input paths in cache keys,
        # causing cache writes beside source WAVs. DirectoryDataset hashes paths.
        ds=DirectoryDataset(str(next(iter(parents))),name=config);ds.wavs=sorted(ds.wavs)
        if {p.resolve() for p in ds.wavs}!={Path(r['source_audio_path']).resolve() for r in rs}:raise ValueError('Directory WAV coverage differs from manifest')
        datasets.append(ds)
    reference=DirectoryDataset(str(a.natural),name='MaSaC_selected_100_stereo_mean')
    reference.wavs=sorted(reference.wavs)
    if len(reference)!=100:raise ValueError('Expected exactly 100 selected natural WAVs')
    write_json(out/'inputs.json',{'baseline_metrics_sha256':digest((a.baseline_run/'tables/metrics.jsonl').read_bytes()),
        'generations':[{k:r[k] for k in ['generation_id','utterance_id','configuration_id','source_audio_path']}|{'audio_sha256':r['audio']['audio_sha256']} for r in rows],
        'natural_sha256':{p.name:digest(p.read_bytes()) for p in reference.wavs},
        'preprocessing':'Official DirectoryDataset for both populations: mono channel mean, librosa resampling 22050 Hz, peak normalization, padding clips below 16000 samples. Noise Tar trims edges without same peak normalization.'})
    torch.set_num_threads(2);np.random.seed(1234);torch.manual_seed(1234)
    torch.cuda.set_per_process_memory_fraction(min(.9,12*1024**3/torch.cuda.get_device_properties(0).total_memory))
    (out/'suite_cache').mkdir()
    # This pinned Lightning checkpoint contains only these additional metadata
    # classes. Keep weights-only loading; never enable unrestricted unpickling.
    from pyannote.audio.core.task import Specifications,Problem,Resolution
    from torch.torch_version import TorchVersion
    checkpoint=a.cache/'models--pyannote--wespeaker-voxceleb-resnet34-LM/snapshots'/PINS['pyannote/wespeaker-voxceleb-resnet34-LM'][0]/'pytorch_model.bin'
    allowed=[Specifications,Problem,Resolution,TorchVersion]
    expected={c.__module__+'.'+c.__name__ for c in allowed}
    actual=set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    if not actual<=expected:raise ValueError('Unexpected checkpoint metadata classes: '+str(actual-expected))
    write_json(out/'safe_checkpoint_metadata.json',{'classes':sorted(actual),'weights_only':True,'scope':'suite construction only'})
    with torch.serialization.safe_globals(allowed):
        suite=BenchmarkSuite(datasets=datasets,reference_datasets=[reference],multilingual=True,include_environment=False,
            skip_errors=False,device='cuda',n_workers=1,cache_dir=str(out/'suite_cache'),write_to_file=str(out/'feature_scores.csv'))
    suite.run();suite.get_aggregated_results().to_csv(out/'aggregate_scores.csv',index=False)
    return {'feature_count':len(suite.benchmarks),'dataset_count':len(datasets),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--natural',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--download-cap',type=int,default=4_000_000_000)
    p.add_argument('--prepared-from',type=Path,help='Reuse hash-verified checkpoints from a prior bounded attempt, with no downloads')
    a=p.parse_args();out=private_directory(a.output)
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['HF_HUB_CACHE']=str(a.cache.resolve())
    os.environ.setdefault('TTSDS_CACHE_DIR',str(out/'feature_cache'))
    if out.exists():raise ValueError('Use a new run directory')
    out.mkdir(parents=True);start=time.time()
    write_json(out/'launch_config.json',{'source_pin':'8ce85ad9e1072a542dc2a6735932a991ebf6e442','multilingual':True,'environment_factor':False,
        'category_weights':{'generic':.25,'prosody':.25,'intelligibility':.25,'speaker':.25},'script_sha256':digest(Path(__file__).read_bytes()),'argv':sys.argv})
    try:
        if a.prepared_from:
            manifest=json.loads((a.prepared_from/'checkpoint_manifest.json').read_text())
            for f in manifest['files']:
                path=a.cache/('models--'+f['model'].replace('/','--'))/'snapshots'/f['revision']/f['file']
                if digest(path.read_bytes())!=f['sha256']:raise ValueError('Changed cached checkpoint/configuration')
            archive=a.prepared_from/'allosaurus_latest.tar.gz'
            if digest(archive.read_bytes())!=manifest['allosaurus']['sha256']:raise ValueError('Changed phone-model archive')
            write_json(out/'checkpoint_manifest.json',manifest)
        else:prefetch(a.cache,out,a.download_cap)
        noise=Path(os.environ['TTSDS_CACHE_DIR'])/'noise-reference'
        if not all((noise/name).is_file() for name in NOISE):
            if a.prepared_from:raise ValueError('Prepared offline noise cache missing; restore pinned noise references first')
            from huggingface_hub import snapshot_download
            snapshot_download('ttsds/noise-reference',repo_type='dataset',revision='bd995bf9352094ea20f1789126bdce16713b501e',local_dir=str(noise),allow_patterns=list(NOISE),token=False)
        for name,expected in NOISE.items():
            if digest((noise/name).read_bytes())!=expected:raise ValueError('Pinned noise reference changed')
        write_json(out/'noise_manifest.json',{'revision':'bd995bf9352094ea20f1789126bdce16713b501e','sha256':NOISE})
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        # The hub constants may already have been imported during prefetch.
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_OFFLINE=True
        result=evaluate(a,out)
        write_json(out/'completion.json',{'status':'complete',**result,'runtime_s':time.time()-start})
    except Exception as e:
        traceback.print_exc()
        write_json(out/'completion.json',{'status':'blocked_attempt','error_type':type(e).__name__,'error':str(e)[:1500],'runtime_s':time.time()-start,'scores_validated':False})
        raise
