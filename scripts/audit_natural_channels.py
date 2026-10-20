"""Read-only stereo-reference diagnostics; does not change channel policy."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from src.evaluation.common import digest,private_directory,write_json,write_jsonl

def audit(source,out):
    out=private_directory(out)
    if out.exists():raise ValueError('Use a new audit directory')
    out.mkdir(parents=True);rows=[]
    for path in sorted(source.glob('*.wav')):
        y,sr=sf.read(path,dtype='float64',always_2d=True)
        if y.shape[1]!=2:raise ValueError('Expected stereo natural reference')
        l,r=y.T;mean=y.mean(axis=1);energies=np.mean(y*y,axis=0)
        combined=np.mean(mean*mean);incoherent=np.sum(energies)/4
        row={'reference_id':path.stem,'audio_sha256':digest(path.read_bytes()),'sample_rate':sr,'duration_s':len(y)/sr,
            'left_right_correlation':float(np.corrcoef(l,r)[0,1]) if np.all(energies>0) else None,
            'left_rms_dbfs':float(10*np.log10(max(energies[0],1e-16))),
            'right_rms_dbfs':float(10*np.log10(max(energies[1],1e-16))),
            'mean_rms_dbfs':float(10*np.log10(max(combined,1e-16))),
            'mean_energy_relative_to_uncorrelated_sum_db':float(10*np.log10(max(combined,1e-16)/max(incoherent,1e-16))),
            'interpretation':'Correlation and energy measure channel mixing sensitivity, not presence of music or speech quality.'}
        rows.append(row)
    write_jsonl(out/'channels.jsonl',rows)
    write_json(out/'completion.json',{'status':'complete','recordings':len(rows),
        'summary':{k:{'median':float(np.median(v)),'min':float(min(v)),'max':float(max(v))} for k in ['left_right_correlation','mean_energy_relative_to_uncorrelated_sum_db'] if (v:=[r[k] for r in rows if r[k] is not None])},
        'channel_policy_changed':False,'script_sha256':digest(Path(__file__).read_bytes())})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.source,a.output)
