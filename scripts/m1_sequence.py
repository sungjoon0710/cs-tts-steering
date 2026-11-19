"""Persistent bounded M1 sequencing: development -> frozen fit -> confirmation.

One GPU, one worker at a time. Stop on technical/resource failures; never run a
confirmation worker before its immutable development gate exists.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','code','python','snapshot','codec-snapshot','existing'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--prior-traces',type=Path)
    p.add_argument('--total-timeout',type=int,default=7200)
    a=p.parse_args();start=time.time();root=a.root
    rows=[json.loads(s) for s in (root/'inputs/cohort.jsonl').read_text().splitlines()]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    jobs=root/'jobs';jobs.mkdir(exist_ok=True)
    state_path=root/'sequence_status.json'
    def state(**kw):
        record={'elapsed_s':time.time()-start,**kw};state_path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    def call(command,name,gpu=True):
        if time.time()-start>a.total_timeout:raise TimeoutError('Total bounded M1 runtime exhausted')
        if gpu:
            # Allow driver cleanup briefly after the previous worker exits.
            ready=False
            for _ in range(15):
                text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                row=next(s for s in text.splitlines() if s.startswith(f'{a.gpu},')).split(', ')
                if int(row[1])<1500 and int(row[2])<6:ready=True;break
                time.sleep(2)
            if not ready:raise RuntimeError(f'GPU {a.gpu} is no longer idle; paused before launching a worker')
        launch=jobs/(name+'.launch.json');log=jobs/(name+'.log')
        if launch.exists() or log.exists():raise FileExistsError('Bounded job name already exists; select a new sequence run')
        launch.write_text(json.dumps({'command':command,'timeout_s':1200,'gpu':a.gpu if gpu else None,'source_code':str(a.code)},indent=2))
        jobenv=env.copy()
        if not gpu:jobenv['CUDA_VISIBLE_DEVICES']=''
        state(status='running',job=name)
        with log.open('x') as stream:
            result=subprocess.run(command,cwd=a.code,env=jobenv,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}; inspect {log}')
    def worker(stage,subset,name):
        if all((root/'study'/r['id']/'subject_complete.json').exists() and not (root/'study'/r['id']/'failure.json').exists() for r in subset):
            return
        cmd=[str(a.python),'-u',str(a.code/'scripts/run_m1.py'),'--cohort',str(root/'inputs/cohort.jsonl'),
            '--protocol',str(root/'inputs/protocol.json'),'--config',str(a.code/'configs/representation_pilot.json'),
            '--snapshot',str(a.snapshot),'--codec-snapshot',str(a.codec_snapshot),'--historical',str(root/'inputs/historical.npz'),
            '--existing',str(a.existing),
            '--output',str(root/'study'),'--stage',stage,'--ids',*[r['id'] for r in subset]]
        if stage=='confirmation':cmd+=['--basis',str(root/'fit/basis.npz')]
        if a.prior_traces:cmd+=['--prior-traces',str(a.prior_traces)]
        call(cmd,name)
    try:
        development=[r for r in rows if r['study_split']=='development']
        for i in range(0,len(development),10):worker('development',development[i:i+10],f'sequence_dev_{i//10+1}')
        if not (root/'fit/confirmation_gate.json').exists():
            call([str(a.python),str(a.code/'scripts/fit_m1.py'),'--root',str(root),'--output',str(root/'fit')],'sequence_fit',False)
        state(status='development_frozen',gate=str(root/'fit/confirmation_gate.json'))
        groups={c:[r for r in rows if r['study_split']=='confirmation' and r['category']==c] for c in sorted({r['category'] for r in rows})}
        ordered=[]
        while any(groups.values()):
            for category in sorted(groups):
                if groups[category]:ordered.append(groups[category].pop(0))
        for i in range(0,len(ordered),10):worker('confirmation',ordered[i:i+10],f'sequence_confirm_{i//10+1}')
        state(status='complete',subjects=len(list((root/'study').glob('*/subject_complete.json'))),confirmation_gate=str(root/'fit/confirmation_gate.json'))
    except Exception as e:
        state(status='stopped',error_type=type(e).__name__,message=str(e));raise


if __name__=='__main__':main()
