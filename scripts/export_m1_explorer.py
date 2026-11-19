"""CPU-only, bounded Model Explorer export of completed M1 captures.

No phase_contrasts, full logits, codebook weights or model checkpoints are copied.
All scalar map/CSV values remain unchanged. Exact vectors are opt-in by subject;
summary-only exports contain zero vector rows, never invented activations.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model_explorer import digest, readjson, token_history_codes
from scripts.verify_m1 import reference_failure


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def safe_name(value):
    if not re.fullmatch('[A-Za-z0-9_-]+',value):
        raise ValueError('Unsafe subject/condition identifier')
    return value


def model_metadata(args):
    if args.model_metadata:
        model=readjson(args.model_metadata)
    else:
        path=args.model_catalog.resolve()
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:
            rows=db.execute('SELECT id,metadata FROM model').fetchall()
            if args.model_id:
                rows=[r for r in rows if r[0]==args.model_id]
            if len(rows)!=1:
                raise ValueError('Select one actual registered model with --model-id')
            model=json.loads(rows[0][1])
        finally:
            db.close()
    for key in ('revision','codec_revision','layers','hidden_size','dtype','model_repo','vendor_inference_sha256'):
        if key not in model:
            raise ValueError('Actual model metadata lacks '+key)
    return model


def export(args):
    root=args.root.resolve();output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new export directory; historical/export data are never overwritten')
    protocol_path=root/'inputs/protocol.json';cohort_path=root/'inputs/cohort.jsonl'
    protocol=readjson(protocol_path);protocol_sha=digest(protocol_path)
    cohort=[json.loads(s) for s in cohort_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    by_id={safe_name(r['id']):r for r in cohort}
    requested=args.ids if args.ids is not None else list(by_id)
    if len(set(requested))!=len(requested) or any(r not in by_id for r in requested):
        raise ValueError('Unknown or duplicate requested subject IDs')
    exact=set(args.exact_ids or [])
    if not exact.issubset(set(requested)):
        raise ValueError('--exact-ids must be among the exported subject IDs')
    if not 0<=args.vector_row_limit<=512:
        raise ValueError('Use a nonnegative bounded vector-row limit')
    unavailable=[]
    for uid in requested:
        if not (root/'study'/uid/'subject_complete.json').exists():
            failed=reference_failure(root,by_id[uid],protocol)
            if not failed:raise ValueError('Subject not completed: '+uid)
            unavailable.append(failed)
    requested=[uid for uid in requested if uid not in {r['id'] for r in unavailable}]
    if exact-{*requested}:raise ValueError('Exact-vector subject has no usable completed trace')
    model=model_metadata(args)
    output.mkdir(parents=True,exist_ok=False)
    sizes=Counter();captures=[];manifests=[]
    for uid in requested:
        row=by_id[uid];subject=root/'study'/uid;subject_done=readjson(subject/'subject_complete.json')
        count=subject_done['histories'] if args.histories=='all' else min(1,subject_done['histories'])
        for index in range(count):
            source=subject/f'h{index}';done=readjson(source/'completion.json');meta=readjson(source/'metadata.json')
            if not done.get('completed') or done['protocol_sha256']!=protocol_sha or meta['protocol_sha256']!=protocol_sha:
                raise ValueError('Incomplete or mismatched protocol: '+uid)
            if digest(source/'metadata.json')!=done['artifacts'].get('metadata.json'):
                raise ValueError('Completed context metadata changed: '+uid)
            if meta['text_sha256']!=row['text_sha256'] or meta['checkpoint']!=model['revision'] or meta['codec']!=model['codec_revision']:
                raise ValueError('Text/checkpoint/codec identity differs: '+uid)
            dest=output/uid/f'h{index}';dest.mkdir(parents=True)
            def ref(path):
                return {'path':path.relative_to(dest).as_posix(),'sha256':digest(path)}
            def copy(name,kind):
                p=source/name
                if name not in done['artifacts'] or digest(p)!=done['artifacts'][name]:
                    raise ValueError('Missing/changed completed artifact: '+str(p))
                target=dest/name;shutil.copyfile(p,target);sizes[kind]+=target.stat().st_size
                return ref(target)
            history_source=subject/f'history_{index}.json';history=readjson(history_source)
            codes=token_history_codes(history)
            if history['text_sha256']!=row['text_sha256'] or history['model_revision']!=model['revision']:
                raise ValueError('Reference-history identity differs')
            history_target=dest/'history.json';shutil.copyfile(history_source,history_target)
            sizes['reference_tokens']+=history_target.stat().st_size
            origin_style=history.get('source_generation',{}).get('style','')
            history_key='reference_'+str(index)
            manifest={'schema_version':1,
                'experiment':{'name':protocol['study'],'dataset':'M1 frozen cohort '+digest(cohort_path)[:16],'model':model},
                'utterance':{'id':uid,'text':row['text'],'metadata':{'language':protocol['language'],
                    'text_sha256':row['text_sha256'],'study_split':row['study_split'],'category':row['category'],
                    'category_evidence':row.get('category_evidence',[]),'source_document':row.get('source_document'),
                    'source':row.get('source'),'source_id':row.get('source_id'),'category_status':'operational text strata, not emotional ground truth'}},
                'conditions':{},'histories':{history_key:{
                    'label':f'H{index} · '+(origin_style or 'default')+' · seed '+str(history['seed']),
                    'recording':None,'tokens':ref(history_target),'metadata':{'source':history.get('source'),
                        'seed':history['seed'],'source_style':origin_style,'source_token_sha256':digest(history_source),
                        'decoder_replayed':False,'waveform_registration':'not part of M1; historical WAVs, where present, remain in their original experiment'}}},
                'contexts':[],'comparisons':[],
                'provenance':{'cohort_sha256':digest(cohort_path),'protocol_sha256':protocol_sha,
                    'source_completion_sha256':digest(source/'completion.json'),
                    'basis_sha256':meta.get('basis_sha256'),'primary_generation_rows':protocol.get('primary_generation_rows'),
                    'source_capture_root':str(source),'export_derived':True}}
            for condition,style in protocol['styles'].items():
                safe_name(condition)
                metadata={'style':style,'speaker':meta['voice'],'language':protocol['language'],
                    'prompt_template':{'name':'Indic-Speak vendor build_prompt','sha256':model['vendor_inference_sha256']},
                    'settings':{'replay':'teacher_forced','probabilities':'raw softmax before sampling warpers',
                        'generation_decoding':protocol['decoding'],'capture_protocol_sha256':protocol_sha},'intervention':None}
                manifest['conditions'][condition]={'label':condition,'metadata':metadata}
                context_ref=copy(condition+'.json','context_json');context=readjson(dest/(condition+'.json'))
                if context['history_tokens']!=codes[:len(context['history_tokens'])] or len(context['positions'])!=meta['n_positions']:
                    raise ValueError('Context/history position association differs')
                npz=source/(condition+'.npz')
                if digest(npz)!=done['artifacts'][condition+'.npz']:
                    raise ValueError('Context tensor hash differs')
                with np.load(npz,allow_pickle=False) as z:
                    saved=z['prediction_rows']
                    keep=np.arange(max(0,len(saved)-args.vector_row_limit),len(saved)) if uid in exact and index==0 else np.array([],dtype=int)
                    for site in ('resid','attn','mlp'):
                        if z[site].shape!=(model['layers'],len(saved),model['hidden_size']):
                            raise ValueError('Context tensor shape differs from actual model')
                    retained={site:z[site][:,keep].copy() for site in ('resid','attn','mlp')}
                    retained['prediction_rows']=saved[keep].copy()
                    retained['final_norm']=z['final_norm'][keep].copy()
                    retained['prompt_embeddings']=z['prompt_embeddings'].copy() if len(keep) else np.empty((0,model['hidden_size']),dtype=z['prompt_embeddings'].dtype)
                    tensor_target=dest/(condition+'.npz');np.savez_compressed(tensor_target,**retained)
                    sizes['selected_vector_npz']+=tensor_target.stat().st_size
                    capture={'subject':uid,'history':index,'condition':condition,'source_vector_rows':len(saved),
                        'exported_vector_rows':len(keep),'kept_prediction_rows':retained['prediction_rows'].tolist(),
                        'source_npz_sha256':digest(npz),'export_npz_sha256':digest(tensor_target)}
                captures.append(capture)
                cm={k:v for k,v in context.items() if k!='top_predictions'}
                cm.setdefault('evidence','observational_replay')
                manifest['contexts'].append({'key':condition,'condition':condition,'history':history_key,
                    'metadata':cm,'arrays':ref(tensor_target),'predictions':context_ref,
                    'capture_levels':{'distance_maps':'all replay positions; M1 primary metrics exclude speech-start j0',
                        'exact_vectors':'selected original rows' if len(keep) else 'not included in this summary export',
                        'local_subset':capture,'attention_weights':'not captured','normalization_inputs':'not captured',
                        'decoder':'not executed for this M1 capture'}})
            for summary in read_csv(source/'comparison_summary.csv'):
                reference=safe_name(summary['reference']);condition=safe_name(summary['condition'])
                prefix=condition+'_minus_'+reference
                maps_ref=copy(prefix+'_heatmaps.npz','scalar_maps_npz')
                predictions_ref=copy(prefix+'_predictions.csv','probability_csv')
                prediction=read_csv(dest/(prefix+'_predictions.csv'))
                with np.load(dest/(prefix+'_heatmaps.npz'),allow_pickle=False) as z:
                    values=z['resid_normalized_l2'];valid=np.isfinite(values)
                    means=np.divide(np.nansum(values,axis=1),valid.sum(axis=1),out=np.full(len(values),np.nan),where=valid.sum(axis=1)>0)
                    peak=int(np.nanargmax(means)) if np.isfinite(means).any() else None
                js=np.array([float(r['jsd_nats']) for r in prediction])
                manifest['comparisons'].append({'a':reference,'b':condition,'arrays':maps_ref,'predictions':predictions_ref,
                    'metadata':{'reference':reference,'condition':condition,'n_positions':len(prediction),
                        'jsd_mean_nats':float(js.mean()),'jsd_median_nats':float(np.median(js)),
                        'argmax_changed_fraction':float(np.mean([float(r['argmax_changed']) for r in prediction])),
                        'residual_peak_layer_mean_l2':peak,'generation_mean_JSD_nats':float(summary['generation_JSD_nats']),
                        'generation_argmax_changed_fraction':float(summary['argmax_changed_fraction']),
                    'summary_scope':'Display means include all replay rows; M1 primary generation metrics exclude speech-start j0',
                    'eos_probability_token_id':128258,'accepted_reference_stop_token_ids':[128258,128001],
                        'history_index':index,'evidence':'observational_replay'}})
            path=dest/'manifest.json';write_json(path,manifest);manifests.append(path.relative_to(output).as_posix())
            sizes['manifest_json']+=path.stat().st_size
    report={'schema_version':1,'source_root':str(root),'requested_cohort_rows':len(cohort),'exported_subjects':requested,
        'unavailable_subjects':unavailable,
        'exported_subject_count':len(requested),'exported_histories':len(manifests),'exported_contexts':len(captures),
        'unavailable_reference_attempts':unavailable,
        'exact_requested_subject_ids':sorted(exact),'exact_vector_contexts':sum(r['exported_vector_rows']>0 for r in captures),
        'exact_subjects_without_saved_vectors':sorted(exact-{r['subject'] for r in captures if r['exported_vector_rows']>0}),
        'vector_row_limit':args.vector_row_limit,'copy_bytes_by_kind':dict(sizes),'uncompressed_export_bytes':sum(sizes.values()),
        'captures':captures,'manifests':manifests,'not_exported':['phase_contrasts','full logits','checkpoint weights','new audio/decoder outputs']}
    write_json(output/'export_summary.json',report)
    if args.archive:
        archive=args.archive.resolve()
        if archive.is_relative_to(output):
            raise ValueError('Place the archive outside the export directory')
        method=zipfile.ZIP_LZMA if args.compression=='lzma' else zipfile.ZIP_DEFLATED
        archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive,'x',compression=method,compresslevel=9 if args.compression=='deflate' else None) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(output).as_posix())
        report['archive_path']=str(archive);report['archive_bytes']=archive.stat().st_size
        report['archive_sha256']=digest(archive)
        report['within_target_compressed_budget']=archive.stat().st_size<=args.max_compressed_mib*1024**2
        write_json(archive.with_suffix(archive.suffix+'.summary.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('captures','manifests','exported_subjects')},ensure_ascii=False))
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--model-metadata',type=Path);group.add_argument('--model-catalog',type=Path)
    p.add_argument('--model-id');p.add_argument('--ids',nargs='+');p.add_argument('--exact-ids',nargs='*',default=[])
    p.add_argument('--histories',choices=['all','primary'],default='all')
    p.add_argument('--vector-row-limit',type=int,default=7)
    p.add_argument('--archive',type=Path);p.add_argument('--compression',choices=['deflate','lzma'],default='deflate')
    p.add_argument('--max-compressed-mib',type=int,default=256)
    return p


if __name__=='__main__':export(parser().parse_args())
