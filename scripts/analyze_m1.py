"""CPU-only frozen M1 analysis. Utterances, never tokens, are inferential units.

No projection is fitted here. Development is exploratory; confirmation uses the
already hashed basis and M2 shortlist. Large phase tensors can remain on Kedi.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.fit_m1 import global_outcomes
from scripts.verify_m1 import reference_failure
from src.evaluation.common import digest,read_jsonl
from src.mechanism_study import SITES,holm

LABELS=('anger','happy','sad')
COLORS={'anger':'#b34b3b','happy':'#b68a20','sad':'#486fa8','news':'#808997','default':'#364b43'}
CATEGORY_NAMES={'emotion_marked':'Emotion-marked','referential_context':'Referential','lexical_control':'Lexical control'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path,rows):
    if not rows:return
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def json_safe(x):
    if isinstance(x,dict):return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [json_safe(v) for v in x]
    if isinstance(x,np.ndarray):return json_safe(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x


def save_json(path,value):
    Path(path).write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def finite_mean(x,axis=None):
    x=np.asarray(x,np.float64);valid=np.isfinite(x)
    count=valid.sum(axis=axis)
    return np.divide(np.where(valid,x,0).sum(axis=axis),count,
                     out=np.full(np.shape(count),np.nan),where=count>0)


def bootstrap_weights(strata,replicates,seed):
    """Within-category resampling; the row/utterance is kept intact across cells."""
    strata=np.asarray(strata)
    if not len(strata) or replicates<20:raise ValueError('Need observations and at least 20 bootstrap draws')
    rng=np.random.default_rng(seed);weights=np.zeros((replicates,len(strata)),np.float64)
    for category in sorted(set(strata)):
        ix=np.flatnonzero(strata==category)
        weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=replicates)
    return weights


def bootstrap_mean(x,strata,replicates=2000,seed=20261116,simultaneous=False):
    """Percentile CIs and optional approximate standardized max-deviation bands.

    Resample independent utterance rows within fixed text-category strata. The
    fitted development basis is held fixed. Missing cells remain undefined; no
    missing metric is imputed as zero. Max bands use the original estimated SE.
    """
    x=np.asarray(x,np.float64)
    if x.ndim==1:x=x[:,None]
    shape=x.shape[1:];matrix=x.reshape(len(x),-1);valid=np.isfinite(matrix)
    count=valid.sum(0);mean=finite_mean(matrix,0)
    centered=np.where(valid,matrix-mean,0)
    variance=np.divide((centered*centered).sum(0),count-1,out=np.full(count.shape,np.nan,dtype=float),where=count>1)
    se=np.sqrt(np.divide(variance,count,out=np.full(variance.shape,np.nan),where=count>0))
    weights=bootstrap_weights(strata,replicates,seed)
    # Matrix multiplication preserves dependence across all cells/features.
    numerator=weights@np.where(valid,matrix,0);denominator=weights@valid.astype(float)
    draws=np.divide(numerator,denominator,out=np.full(numerator.shape,np.nan),where=denominator>0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        lo,hi=np.nanquantile(draws,[.025,.975],axis=0)
    out={'mean':mean.reshape(shape),'lower':lo.reshape(shape),'upper':hi.reshape(shape),'n':count.reshape(shape),'se':se.reshape(shape)}
    if simultaneous:
        eligible=(count>1)&np.isfinite(se)&(se>8*np.finfo(float).eps*np.maximum(1,np.abs(mean)))
        if eligible.any():
            maximum=np.max(np.abs((draws[:,eligible]-mean[eligible])/se[eligible]),axis=1)
            critical=float(np.quantile(maximum,.95))
        else:critical=np.nan
        out.update(simultaneous_lower=(mean-critical*se).reshape(shape),
                   simultaneous_upper=(mean+critical*se).reshape(shape),max_deviation_critical=critical,
                   simultaneous_valid_cells=int(eligible.sum()))
    return out


def primary_tests(rows,outcomes,replicates,seed,requested_confirmation,requested_histories):
    tests=[];pvalues=[]
    for index,name in enumerate(outcomes):
        chosen=[r for r in rows if r['study_split']=='confirmation' and np.isfinite(float(r.get(name,np.nan)))]
        x=np.array([r[name] for r in chosen],float);strata=[r['category'] for r in chosen]
        status='tested';raw_p=None;t=None
        if len(x)<2:status='insufficient observations'
        elif np.std(x,ddof=1)<=8*np.finfo(float).eps*max(1,abs(x.mean())):status='numerically degenerate sample; no variance-based test'
        else:
            test=stats.ttest_1samp(x,0,alternative='two-sided')
            t=float(test.statistic);raw_p=float(test.pvalue)
        pvalues.append(raw_p if raw_p is not None else 1.)
        interval=bootstrap_mean(x,strata,replicates,seed+index) if len(x) else None
        tests.append({'outcome':name,'n_requested':requested_histories if name=='history_direction_cosine' else requested_confirmation,
            'n':len(x),'mean':float(x.mean()) if len(x) else np.nan,'median':float(np.median(x)) if len(x) else np.nan,
            'sd':float(np.std(x,ddof=1)) if len(x)>1 else np.nan,
            'ci_lower':float(interval['lower'][0]) if interval else np.nan,'ci_upper':float(interval['upper'][0]) if interval else np.nan,
            'positive':int((x>0).sum()),'negative':int((x<0).sum()),'zero':int((x==0).sum()),
            't':t,'p_raw':raw_p,'status':status,'category_counts':json.dumps({c:strata.count(c) for c in sorted(set(strata))})})
    for row,p in zip(tests,holm(pvalues)):
        row['p_holm']=float(p)
        row['reject_at_05']=bool(p<.05 and row['status']=='tested')
    return tests


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:v.copy() for k,v in z.items()}


def verify_transfer_receipt(root):
    receipt=root/'transfer_manifest.json'
    if not receipt.exists():return {'present':False,'verified_files':0}
    entries=read_json(receipt)['sha256']
    for relative,expected in entries.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Unsafe or missing transfer receipt path: '+relative)
        if digest(path.read_bytes())!=expected:raise ValueError('Transfer receipt mismatch: '+relative)
    return {'present':True,'verified_files':len(entries),'receipt_sha256':digest(receipt.read_bytes())}


def verify_gate(root):
    gate=read_json(root/'fit/confirmation_gate.json')
    checks=[('basis_sha256',root/'fit/basis.npz'),('shortlist_sha256',root/'fit/M2_shortlist.json'),
            ('cohort_sha256',root/'inputs/cohort.jsonl'),('protocol_sha256',root/'inputs/protocol.json')]
    for key,path in checks:
        if digest(path.read_bytes())!=gate[key]:raise ValueError('Frozen gate mismatch: '+key)
    if gate.get('fitted_on_confirmation') is not False:raise ValueError('No verified development-only fit')
    cohort=read_jsonl(root/'inputs/cohort.jsonl');development={r['id'] for r in cohort if r['study_split']=='development'}
    if not set(gate['development_ids'])<=development:raise ValueError('Non-development IDs in basis fit')
    return gate,cohort


def projection_missingness(geometry,basis):
    """Mark undefined directions in analysis copies, retaining original files.

    With nonzero shared/specific templates, zero shared cosine plus undefined
    specific cosine implies a zero normalized input contrast in the current
    projector. A genuine orthogonal nonzero vector has finite specific cosine.
    """
    g={k:v.copy() for k,v in geometry.items()}
    shared_valid=np.isfinite(basis['shared']).all(-1)&(np.linalg.norm(basis['shared'],axis=-1)>0)
    specific_valid=np.isfinite(basis['specific']).all(-1)&(np.linalg.norm(basis['specific'],axis=-1)>0)
    undefined_shared=np.broadcast_to(~shared_valid,g['shared_cosine'].shape).copy()
    zero_input=(g['shared_cosine']==0)&~np.isfinite(g.get('specific_cosine',g['specific_margin']))&specific_valid&shared_valid
    undefined_shared|=zero_input
    g['shared_cosine'][undefined_shared]=np.nan
    if 'shared_energy_fraction' in g:g['shared_energy_fraction'][undefined_shared]=np.nan
    # Own-minus-two-other templates requires all three style templates.
    undefined_margin=np.broadcast_to(~np.all(specific_valid,axis=0),g['specific_margin'].shape)|zero_input
    g['specific_margin'][undefined_margin]=np.nan
    return g,{'undefined_shared_cells':int(undefined_shared.sum()),'zero_input_candidate_cells':int(zero_input.sum()),
              'undefined_specificity_cells':int(undefined_margin.sum()),'total_style_site_layer_phase_cells':g['shared_cosine'].size}


def tie_description(score,selected_site,selected_layer):
    score=np.asarray(score)
    maximum=np.nanmax(score);ties=np.flatnonzero(score.ravel()==maximum)
    chosen=int(np.ravel_multi_index((selected_site,selected_layer),score.shape))
    return {'maximum':float(maximum),'exact_tie_flat_indices':ties.tolist(),'exact_tie_count':len(ties),
            'first_flat_index':int(ties[0]),'frozen_flat_index':chosen,'matches_first_flat_rule':chosen==int(ties[0])}


def selection_tie_audit(root,gate,shortlist):
    # Read the original development scores, before any undefined-cell annotation.
    dev=[load_npz(root/'fit'/(uid+'_geometry.npz')) for uid in gate['development_ids']]
    result={}
    for name,metric in [('shared_structure','shared_cosine'),('style_specific_structure','specific_margin')]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            score=np.nanmean(np.stack([g[metric] for g in dev]),axis=(0,1,4))
        chosen=shortlist[name]
        result[name]=tie_description(score,SITES.index(chosen['site']),chosen['layer'])
    means=[]
    for p in range(7):
        means.append(float(finite_mean([float(r['jsd_nats']) for uid in gate['development_ids']
            for r in csv_rows(root/'study'/uid/'h0/probability_phase_metrics.csv') if r['reference']=='news' and int(r['phase'])==p])))
    ties=np.flatnonzero(np.asarray(means)==max(means)).tolist();chosen=shortlist['prediction_phase']['phase']
    result['prediction_phase']={'development_phase_means':means,'exact_tie_indices':ties,'first_index':ties[0],
        'frozen_index':chosen,'matches_first_flat_rule':chosen==ties[0]}
    result['selection_rule']='Exact ties resolved by first flat index; frozen choices audited, never reselected.'
    return result


def collect(root,gate,cohort,allow_partial=False,allow_missing_tensors=False):
    records=[];coverage=[];verified=[];omitted=[];geometries={};positions=[];probability=[];components=[];history_pairs=[];projection_audit=[];grammar=[];grammar_exceptions=[]
    protocol_sha=gate['protocol_sha256']
    protocol=read_json(root/'inputs/protocol.json');basis=load_npz(root/'fit/basis.npz')
    checkpoints=set()
    for entry in cohort:
        subject=root/'study'/entry['id'];complete=subject/'subject_complete.json';failure=subject/'failure.json'
        if not complete.exists():
            status='failed' if failure.exists() else 'unattempted'
            if status=='failed':reference_failure(root,entry,protocol)
            coverage.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],'status':status,
                             'requested_histories':2 if entry['second_history'] else 1,'completed_histories':0,
                             'error':read_json(failure).get('message','') if failure.exists() else ''})
            if status=='unattempted' and not allow_partial:raise ValueError('Unattempted study subject '+entry['id'])
            continue
        count=read_json(complete)['histories']
        if count not in (1,2) or (count==2 and not entry['second_history']):
            raise ValueError('Unexpected completed history count: '+entry['id'])
        if entry['second_history'] and (count!=2 or not (subject/'history_robustness.npz').is_file() or not (subject/'history_comparison.json').is_file()) and not allow_partial:
            raise ValueError('Requested second-history robustness is incomplete: '+entry['id'])
        coverage.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],
            'status':'complete','requested_histories':2 if entry['second_history'] else 1,'completed_histories':count,'error':''})
        for history in range(count):
            directory=subject/f'h{history}';done=read_json(directory/'completion.json');meta=read_json(directory/'metadata.json')
            if done['protocol_sha256']!=protocol_sha or meta['protocol_sha256']!=protocol_sha:raise ValueError('Per-history protocol changed')
            if entry['study_split']=='confirmation' and meta['basis_sha256']!=gate['basis_sha256']:raise ValueError('Confirmation used another basis')
            if meta['text_sha256']!=entry['text_sha256']:raise ValueError('Text association changed')
            for key,expected in [('id',entry['id']),('study_split',entry['study_split']),('category',entry['category']),('history_index',history)]:
                if meta.get(key)!=expected:raise ValueError('Per-history association differs: '+key)
            if meta.get('voice')!=protocol['voice'] or meta.get('n_positions')!=done['positions']:
                raise ValueError('Voice/position coverage differs')
            checkpoints.add((meta['checkpoint'],meta['codec']))
            if len(checkpoints)>1:raise ValueError('Mixed checkpoint/codec traces')
            for name,sha in done['artifacts'].items():
                p=directory/name
                if not p.exists():
                    if allow_missing_tensors and p.suffix=='.npz' and name not in ('geometry_metrics.npz','historical_alignment_positions.npz'):
                        omitted.append(str(p));continue
                    raise ValueError('Missing completed artifact '+str(p))
                if digest(p.read_bytes())!=sha:raise ValueError('Changed artifact '+str(p))
                verified.append(str(p))
            n=done['positions'];full_codes=len(meta['history_provenance']['generated_token_ids'])-1
            if n<14 or n%7 or n>full_codes:raise ValueError('Invalid complete-frame prediction grid')
            if history==0:
                path=root/'fit'/(entry['id']+'_geometry.npz') if entry['study_split']=='development' else directory/'geometry_metrics.npz'
                geometry,audit=projection_missingness(load_npz(path),basis)
                projection_audit.append({'id':entry['id'],'study_split':entry['study_split'],**audit})
                geometries[entry['id']]=geometry
                outcomes=global_outcomes(subject,geometry)
                records.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],
                    'text_sha256':entry['text_sha256'],'source_document':entry['source_document'],'n_positions':n,
                    'full_history_codes':full_codes,'inspected_fraction':n/full_codes,**outcomes})
            for row in csv_rows(directory/'probability_phase_metrics.csv'):
                probability.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],'history':history,
                    **{k:float(v) if k not in ('reference','condition') else v for k,v in row.items()}})
            if history==0:
                for row in csv_rows(directory/'residual_decomposition.csv'):
                    components.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],
                        **{k:float(v) if k!='condition' else v for k,v in row.items()}})
            positional_file=directory/'historical_alignment_positions.npz'
            positional=load_npz(positional_file)
            for reference,values in positional.items():
                if reference not in ('news','default') or values.shape!=(3,28,n):
                    raise ValueError('Unexpected historical position tensor')
                frames=np.arange(n)//7;tertile=np.minimum(2,3*frames//(n//7));mask=np.arange(n)>0
                for label in range(3):
                    for third in range(3):
                        selected=mask&(tertile==third)
                        # Per-position cosine mean is a separate secondary
                        # statistic from cosine of the phase-averaged vector.
                        positions.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],'history':history,
                            'reference':reference,'condition':LABELS[label],'tertile':third,'n_positions':int(selected.sum()),
                            'mean_pointwise_historical_cosine':float(finite_mean(values[label,:,selected]))})
            # Probability chronology from saved scalar rows, never model replay.
            for condition in LABELS:
                p=directory/(condition+'_minus_news_predictions.csv')
                rows=csv_rows(p)
                valid=[r for r in rows if int(r['j'])>0]
                if [int(r['j']) for r in rows]!=list(range(n)):
                    raise ValueError('Missing/duplicated prediction rows: '+str(p))
                if any('predicted_phase' in r and int(r['predicted_phase'])!=int(r['j'])%7 for r in rows):
                    raise ValueError('Prediction phase metadata disagrees with complete frame grid')
                phase_mean=lambda key:float(finite_mean([finite_mean([float(r[key]) for r in valid if int(r['j'])%7==phase]) for phase in range(7)])) if key in rows[0] else np.nan
                gr={'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],
                    'history':history,'condition':condition,'reference':'news','n_positions':len(valid)}
                for key in ('jsd_nats','phase_jsd_nats','reference_phase_mass','changed_phase_mass','reference_eos_probability','changed_eos_probability'):
                    gr[key]=phase_mean(key)
                gr['full_minus_conditional_JSD']=gr['jsd_nats']-gr['phase_jsd_nats']
                for side in ('reference','changed'):
                    mass=side+'_phase_mass';eos=side+'_eos_probability'
                    gr['min_'+mass]=min(float(r[mass]) for r in valid) if mass in rows[0] else np.nan
                    gr['max_'+eos]=max(float(r[eos]) for r in valid) if eos in rows[0] else np.nan
                    gr[side+'_positions_below_099_phase_mass']=sum(float(r[mass])<.99 for r in valid) if mass in rows[0] else np.nan
                    gr[side+'_positions_above_001_EOS']=sum(float(r[eos])>.01 for r in valid) if eos in rows[0] else np.nan
                grammar.append(gr)
                for r in valid:
                    if ('reference_phase_mass' in r and (min(float(r['reference_phase_mass']),float(r['changed_phase_mass']))<.99 or max(float(r['reference_eos_probability']),float(r['changed_eos_probability']))>.01)):
                        grammar_exceptions.append({'id':entry['id'],'study_split':entry['study_split'],'history':history,'condition':condition,**r})
                for third in range(3):
                    selected=[r for r in rows if int(r['j'])>0 and min(2,3*(int(r['j'])//7)//(n//7))==third]
                    positions.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],'history':history,
                        'reference':'news_probability','condition':condition,'tertile':third,'n_positions':len(selected),
                        'mean_pointwise_historical_cosine':np.nan,
                        'mean_JSD_nats':float(finite_mean([float(r['jsd_nats']) for r in selected]))})
        if count==2:
            comparison=read_json(subject/'history_comparison.json') if (subject/'history_comparison.json').exists() else {}
            a=[r['jsd_nats'] for r in probability if r['id']==entry['id'] and r['history']==0 and r['reference']=='news']
            b=[r['jsd_nats'] for r in probability if r['id']==entry['id'] and r['history']==1 and r['reference']=='news']
            record=next(r for r in records if r['id']==entry['id'])
            history_pairs.append({'id':entry['id'],'study_split':entry['study_split'],'category':entry['category'],
                'JSD_history0':float(finite_mean(a)),'JSD_history1':float(finite_mean(b)),
                'paired_JSD_difference':float(finite_mean(b)-finite_mean(a)),
                'history_direction_cosine':record['history_direction_cosine'],
                'identical_inspected_prefix':comparison.get('identical_inspected_prefix',False)})
    # CSV schema is explicit even when a supplementary probability field is absent.
    for row in positions:row.setdefault('mean_JSD_nats',np.nan)
    return {'records':records,'coverage':coverage,'verified':verified,'omitted':omitted,'geometry':geometries,
            'positions':positions,'probability':probability,'components':components,'history_pairs':history_pairs,'projection_audit':projection_audit,
            'token_grammar':grammar,'token_grammar_exceptions':grammar_exceptions}


def summarize_cells(rows,keys,metric,replicates,seed):
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    result=[]
    for i,(key,items) in enumerate(sorted(groups.items())):
        # Callers must already have one value per utterance in a cell.
        if len({r['id'] for r in items})!=len(items):raise ValueError('Repeated utterance treated as independent')
        x=[r[metric] for r in items]
        b=bootstrap_mean(x,[r['category'] for r in items],replicates,seed+i)
        result.append({**dict(zip(keys,key)),'metric':metric,'n':int(b['n'][0]),'mean':float(b['mean'][0]),
                       'ci_lower':float(b['lower'][0]),'ci_upper':float(b['upper'][0])})
    return result


def make_figures(output,data,curves,phase_summary,category_summary,history_summary,map_summaries,component_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    site_colors={'resid':'#246aa5','attn':'#b86b2d','mlp':'#4f865b'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for j,metric in enumerate(('shared_cosine','specific_margin')):
        for site,color in site_colors.items():
            for split,style in [('confirmation','-'),('development','--')]:
                rows=[r for r in curves if r['metric']==metric and r['site']==site and r['study_split']==split]
                if not rows:continue
                x=[r['layer'] for r in rows];y=[r['mean'] for r in rows]
                axes[j].plot(x,y,style,color=color,label=site+' / '+split)
                if split=='confirmation':axes[j].fill_between(x,[r['lower'] for r in rows],[r['upper'] for r in rows],color=color,alpha=.13)
        axes[j].axhline(0,color='black',lw=.7);axes[j].set(xlabel='Zero-based layer',ylabel='Utterance mean cosine' if j==0 else 'Own minus other-style cosine',title='Shared consensus transfer' if j==0 else 'Residualized style specificity');axes[j].grid(alpha=.15)
    axes[0].legend(fontsize=8,ncol=2);fig.suptitle('News-relative geometry · development fit frozen before confirmation\nShading: 95% pointwise stratified utterance-bootstrap intervals',fontsize=11)
    fig.savefig(output/'01_shared_vs_specific.png',dpi=180);plt.close(fig)

    hist=map_summaries['historical_news'];fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for k,label in enumerate(LABELS):
        im=axes[0,k].imshow(hist['mean'][k],origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='coolwarm')
        axes[0,k].set(title=label+' − News',xlabel='Predicted SNAC phase',ylabel='Zero-based layer');axes[0,k].set_xticks(range(7));fig.colorbar(im,ax=axes[0,k],shrink=.7,label='Cosine to fixed historical anchor direction')
        for ref,ls in [('historical_news','-'),('historical_default','--')]:
            m=map_summaries[ref]
            axes[1,k].plot(range(28),np.nanmean(m['mean'][k],axis=1),ls,color=COLORS[label],label='News-relative' if ref.endswith('news') else 'Default-relative (secondary)')
        axes[1,k].axhline(0,color='black',lw=.7);axes[1,k].set(xlabel='Zero-based layer',ylabel='Phase-balanced alignment');axes[1,k].grid(alpha=.15);axes[1,k].legend(fontsize=8)
    fig.suptitle('Historical speech-start direction → generation-time phase-mean contrasts\nConfirmation only; j=0 excluded; no causal or perceptual emotion claim',fontsize=11)
    fig.savefig(output/'02_anchor_to_generation.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
    for term,color in zip(('incoming','attention','mlp','rounding'),('#556d86','#b86b2d','#4f865b','#a05080')):
        rows=[r for r in component_summary if r['term']==term]
        ax.plot([r['layer'] for r in rows],[r['mean'] for r in rows],label=term,color=color)
        ax.fill_between([r['layer'] for r in rows],[r['lower'] for r in rows],[r['upper'] for r in rows],color=color,alpha=.1)
    ax.axhline(0,color='black',lw=.6);ax.axhline(1,color='#888',lw=.5,ls=':');ax.set(xlabel='Zero-based layer',ylabel='Signed projection onto Δ post-block residual',title='Actual residual arithmetic: incoming + attention + MLP + rounding remainder')
    ax.legend(ncol=4);ax.grid(alpha=.15);fig.text(.5,-.01,'Observational directional decomposition; terms may exceed one or cancel. Ribbons: pointwise utterance-bootstrap CIs, not causal fractions.',ha='center',fontsize=9)
    fig.savefig(output/'03_component_propagation.png',dpi=180,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for j,(ref,metric,title) in enumerate([('news','jsd_nats','Full-vocabulary JSD: News-relative'),('news','phase_jsd_nats','Phase-conditional JSD: News-relative'),('default','jsd_nats','Full JSD: Default-relative (secondary)')]):
        for label in (LABELS if ref=='news' else LABELS+('news',)):
            rs=[r for r in phase_summary if r['reference']==ref and r['condition']==label and r['metric']==metric]
            if not rs:continue
            axes[j].plot([r['phase'] for r in rs],[r['mean'] for r in rs],marker='o',color=COLORS[label],label=label)
            axes[j].fill_between([r['phase'] for r in rs],[r['ci_lower'] for r in rs],[r['ci_upper'] for r in rs],color=COLORS[label],alpha=.13)
        axes[j].set(title=title,xlabel='Predicted SNAC phase',ylabel='JSD (nats)');axes[j].set_xticks(range(7));axes[j].grid(alpha=.15);axes[j].legend(fontsize=8)
    fig.suptitle('Final-LM raw probabilities · fixed audio history within each text\n95% utterance-bootstrap intervals; Default lacks the four-token style block',fontsize=11)
    fig.savefig(output/'04_token_predictions_by_phase.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    categories=sorted({r['category'] for r in category_summary})
    for j,metric in enumerate(('shared_alignment_resid','specificity_margin_resid','historical_news_anchor_alignment')):
        rows=[r for r in category_summary if r['metric']==metric]
        for i,row in enumerate(rows):
            y=categories.index(row['category'])+(j-1)*.17
            color=list(site_colors.values())[j]
            axes[0].plot([row['ci_lower'],row['ci_upper']],[y,y],color=color)
            axes[0].plot(row['mean'],y,'o',color=color,label=metric if i==0 else None)
    axes[0].set(yticks=range(len(categories)),yticklabels=[CATEGORY_NAMES[c] for c in categories],xlabel='Utterance mean diagnostic',title='Operational text strata');axes[0].axvline(0,color='black',lw=.6);axes[0].legend(fontsize=7)
    pairs=[r for r in data['history_pairs'] if r['study_split']=='confirmation']
    for c,color in zip(categories,('#a24f44','#3c78a5','#63965d')):
        rows=[r for r in pairs if r['category']==c and not r['identical_inspected_prefix']]
        axes[1].scatter([r['JSD_history0'] for r in rows],[r['JSD_history1'] for r in rows],s=23,color=color,label=CATEGORY_NAMES[c])
    if pairs:
        values=[r[k] for r in pairs for k in ('JSD_history0','JSD_history1')];lo,hi=min(values),max(values)
        axes[1].plot([lo,hi],[lo,hi],color='#777',lw=.8,ls='--')
    axes[1].set(xlabel='History 0 mean JSD (nats)',ylabel='History 1 mean JSD (nats)',title='Same texts, different reference histories');axes[1].legend(fontsize=8)
    for i,row in enumerate(history_summary):
        axes[2].plot([row['ci_lower'],row['ci_upper']],[i,i],color='#3c78a5');axes[2].plot(row['mean'],i,'o',color='#3c78a5')
    axes[2].set(yticks=range(len(history_summary)),yticklabels=[CATEGORY_NAMES[r['category']]+' (n='+str(r['n'])+')' for r in history_summary],xlabel='Mean cosine between history-specific style contrasts',title='Direction robustness (paired utterances)');axes[2].axvline(0,color='black',lw=.6)
    fig.suptitle('Confirmation text/history robustness · dependent code positions never count as independent samples',fontsize=11)
    fig.savefig(output/'05_text_history_robustness.png',dpi=180);plt.close(fig)


def build(args):
    root=args.root.resolve();transport=verify_transfer_receipt(root)
    gate,cohort=verify_gate(root);protocol=read_json(root/'inputs/protocol.json')
    if args.bootstrap and args.bootstrap!=protocol['bootstrap']['replicates'] and not args.allow_partial:
        raise ValueError('A non-protocol bootstrap count requires explicit --allow-partial engineering mode')
    reps=args.bootstrap or protocol['bootstrap']['replicates'];seed=protocol['seed_for_analysis']
    data=collect(root,gate,cohort,args.allow_partial,args.allow_missing_tensors)
    confirmation=[r for r in data['records'] if r['study_split']=='confirmation']
    if not confirmation:raise ValueError('No confirmation records; do not fabricate confirmation analysis')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    requested=sum(r['study_split']=='confirmation' for r in cohort)
    requested_h=sum(r['study_split']=='confirmation' and r['second_history'] for r in cohort)
    tests=primary_tests(data['records'],protocol['primary_outcomes'],reps,seed,requested,requested_h)
    save_csv(out/'primary_confirmation_tests.csv',tests)
    save_csv(out/'projection_missingness_audit.csv',data['projection_audit'])
    save_csv(out/'per_utterance_outcomes.csv',data['records']);save_csv(out/'coverage.csv',data['coverage'])
    save_csv(out/'position_tertiles.csv',data['positions']);save_csv(out/'probability_per_utterance_phase.csv',data['probability'])
    save_csv(out/'component_decomposition_per_utterance.csv',data['components']);save_csv(out/'paired_history_results.csv',data['history_pairs'])
    save_csv(out/'token_grammar_per_utterance.csv',data['token_grammar'])
    save_csv(out/'token_grammar_exceptions.csv',data['token_grammar_exceptions'])
    grammar_summary=[]
    for metric in ('jsd_nats','phase_jsd_nats','full_minus_conditional_JSD','reference_phase_mass','changed_phase_mass','reference_eos_probability','changed_eos_probability'):
        selected=[r for r in data['token_grammar'] if r['study_split']=='confirmation' and r['history']==0]
        grammar_summary.extend(summarize_cells(selected,('condition',),metric,reps,seed+50))
    save_csv(out/'token_grammar_summary.csv',grammar_summary)
    # No projection fit here: these are confirmation applications of frozen axes.
    map_summaries={};maps={k:np.stack([data['geometry'][r['id']][k] for r in confirmation]) for k in
        ('historical_news','historical_default','shared_cosine','specific_margin')}
    strata=[r['category'] for r in confirmation]
    for i,(key,x) in enumerate(maps.items()):
        b=bootstrap_mean(x,strata,reps,seed+100+i,simultaneous=True);map_summaries[key]=b
        np.savez_compressed(out/(key+'_confirmation_bands.npz'),**b)
        rows=[]
        for ix in np.ndindex(b['mean'].shape):
            descriptors={'condition':LABELS[ix[0]],'layer':ix[-2],'phase':ix[-1]}
            if len(ix)==4:descriptors['site']=SITES[ix[1]]
            rows.append({**descriptors,**{name:float(b[name][ix]) for name in ('mean','lower','upper','simultaneous_lower','simultaneous_upper','n')}})
        save_csv(out/(key+'_confirmation_bands.csv'),rows)
    curve_rows=[]
    for split in ('development','confirmation'):
        selected=[r for r in data['records'] if r['study_split']==split]
        if not selected:continue
        for metric in ('shared_cosine','specific_margin'):
            x=np.stack([finite_mean(data['geometry'][r['id']][metric],axis=(0,3)) for r in selected]) # N x site x layer
            b=bootstrap_mean(x,[r['category'] for r in selected],reps,seed+200,simultaneous=True)
            for site in range(3):
                for layer in range(28):curve_rows.append({'study_split':split,'metric':metric,'site':SITES[site],'layer':layer,
                    **{k:float(b[k][site,layer]) for k in ('mean','lower','upper','n')}})
    save_csv(out/'geometry_layer_curves.csv',curve_rows)
    phase_rows=[r for r in data['probability'] if r['study_split']=='confirmation' and r['history']==0]
    phase_summary=[]
    for metric in ('jsd_nats','phase_jsd_nats','delta_target_logp','argmax_changed'):
        phase_summary.extend(summarize_cells(phase_rows,('reference','condition','phase'),metric,reps,seed+300))
    save_csv(out/'probability_phase_summary.csv',phase_summary)
    category_summary=[]
    for metric in ('shared_alignment_resid','specificity_margin_resid','historical_news_anchor_alignment','news_relative_JSD'):
        category_summary.extend(summarize_cells(confirmation,('category',),metric,reps,seed+400))
    save_csv(out/'text_category_summary.csv',category_summary)
    history_rows=[r for r in data['history_pairs'] if r['study_split']=='confirmation' and not r['identical_inspected_prefix']]
    history_summary=summarize_cells(history_rows,('category',),'history_direction_cosine',reps,seed+500) if history_rows else []
    save_csv(out/'history_direction_summary.csv',history_summary)
    history_effects=[]
    for metric in ('paired_JSD_difference','history_direction_cosine'):
        if history_rows:
            b=bootstrap_mean([r[metric] for r in history_rows],[r['category'] for r in history_rows],reps,seed+501)
            history_effects.append({'metric':metric,'n':int(b['n'][0]),'mean':float(b['mean'][0]),
                'ci_lower':float(b['lower'][0]),'ci_upper':float(b['upper'][0]),'comparison':'secondary minus primary history, identical prefixes excluded'})
    save_csv(out/'paired_history_effect_summary.csv',history_effects)
    tertile_summary=[]
    for metric in ('mean_pointwise_historical_cosine','mean_JSD_nats'):
        selected=[r for r in data['positions'] if r['study_split']=='confirmation' and r['history']==0 and np.isfinite(r[metric])]
        tertile_summary.extend(summarize_cells(selected,('reference','condition','tertile'),metric,reps,seed+520))
    save_csv(out/'position_tertile_summary.csv',tertile_summary)
    component_terms=('incoming','attention','mlp','rounding')
    component_x=[]
    for r in confirmation:
        selected=[x for x in data['components'] if x['id']==r['id']]
        component_x.append([[finite_mean([x[term] for x in selected if int(x['layer'])==layer]) for layer in range(28)] for term in component_terms])
    component_bands=bootstrap_mean(component_x,strata,reps,seed+550,simultaneous=True)
    component_summary=[{'term':term,'layer':layer,**{key:float(component_bands[key][i,layer]) for key in ('mean','lower','upper','simultaneous_lower','simultaneous_upper','n')}}
                       for i,term in enumerate(component_terms) for layer in range(28)]
    save_csv(out/'component_propagation_summary.csv',component_summary)
    shortlist=read_json(root/'fit/M2_shortlist.json');shortlist_rows=[]
    ties=selection_tie_audit(root,gate,shortlist)
    save_json(out/'development_selection_tie_audit.json',ties)
    if any(not ties[k]['matches_first_flat_rule'] for k in ('shared_structure','style_specific_structure','prediction_phase')):
        raise ValueError('Frozen M2 choices do not reproduce the declared development-only tie rule')
    for key,metric in [('shared_structure','shared_cosine'),('style_specific_structure','specific_margin')]:
        chosen=shortlist[key];site=SITES.index(chosen['site']);layer=chosen['layer']
        rows=[{'id':r['id'],'category':r['category'],'value':float(finite_mean(data['geometry'][r['id']][metric][:,site,layer]))} for r in confirmation]
        b=bootstrap_mean([r['value'] for r in rows],[r['category'] for r in rows],reps,seed+600)
        shortlist_rows.append({'hypothesis':key,'site':chosen['site'],'layer':layer,'development_value':chosen['development_mean'],
            'confirmation_mean':float(b['mean'][0]),'ci_lower':float(b['lower'][0]),'ci_upper':float(b['upper'][0]),'n':len(rows),
            'selection':'frozen development choice; not retuned'})
    save_csv(out/'frozen_M2_shortlist_confirmation.csv',shortlist_rows)
    chosen_phase=shortlist['prediction_phase']['phase'];phase_effects=[]
    for r in confirmation:
        rows=[p for p in phase_rows if p['id']==r['id'] and p['reference']=='news']
        at=finite_mean([p['jsd_nats'] for p in rows if int(p['phase'])==chosen_phase])
        elsewhere=finite_mean([p['jsd_nats'] for p in rows if int(p['phase'])!=chosen_phase])
        phase_effects.append({'id':r['id'],'category':r['category'],'frozen_phase':chosen_phase,
                             'selected_phase_JSD':float(at),'selected_minus_other_phase_JSD':float(at-elsewhere)})
    save_csv(out/'frozen_M2_phase_per_utterance.csv',phase_effects)
    phase_validation=[]
    for metric in ('selected_phase_JSD','selected_minus_other_phase_JSD'):
        b=bootstrap_mean([r[metric] for r in phase_effects],strata,reps,seed+610)
        phase_validation.append({'frozen_phase':chosen_phase,'metric':metric,'n':int(b['n'][0]),
            'mean':float(b['mean'][0]),'ci_lower':float(b['lower'][0]),'ci_upper':float(b['upper'][0]),
            'selection':'fixed on development; descriptive confirmation CI, outside fixed ten tests'})
    save_csv(out/'frozen_M2_phase_confirmation.csv',phase_validation)
    make_figures(out,data,curve_rows,phase_summary,category_summary,history_summary,map_summaries,component_summary)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'analysis_code_sha256':digest(Path(__file__).read_bytes()),
        'protocol_sha256':gate['protocol_sha256'],'cohort_sha256':gate['cohort_sha256'],'basis_sha256':gate['basis_sha256'],
        'shortlist_sha256':gate['shortlist_sha256'],'bootstrap_replicates':reps,'seed':seed,'allow_partial':args.allow_partial,
        'partial_results':bool(args.allow_partial or any(r['status']=='unattempted' for r in data['coverage'])),
        'verified_available_artifacts':len(data['verified']),'omitted_tensor_artifacts':data['omitted'],
        'requested_confirmation':requested,'completed_confirmation':len(confirmation),'requested_secondary_histories':requested_h,
        'complete_paired_confirmation':len(history_rows),'basis_refitted':False,'causal_interventions':0}
    manifest['transfer_receipt']=transport
    save_json(out/'analysis_manifest.json',manifest)
    report(out,manifest,tests,shortlist_rows,data,gate,protocol)
    print(json.dumps(json_safe({'output':str(out),'confirmation_n':len(confirmation),'paired_history_n':len(history_rows),'primary_tests':tests})))


def report(out,manifest,tests,shortlist_rows,data,gate,protocol):
    lines=['# M1 multi-style generation-time mechanism study','',
        '**INTERIM — incomplete requested confirmation.**' if manifest['partial_results'] else '**Frozen confirmation analysis.**','',
        f"Requested {manifest['requested_confirmation']} confirmation utterances; {manifest['completed_confirmation']} contributed complete primary metrics. Paired-history confirmation: {manifest['complete_paired_confirmation']}/{manifest['requested_secondary_histories']}.",
        '','## What was tested','',
        'Five prompts (empty, ANGER, happy, sad, news) on the same valid code history within each utterance. All 28 layers and seven SNAC phases were observed. Primary generation summaries exclude speech-start prediction j=0 and weight phases equally. News-relative contrasts control style-block format; they do not establish emotional semantics. Default comparisons are secondary because its style block is absent.',
        '',f"The rank-one shared consensus and residualized style means were fitted only on {len(gate['development_ids'])} development texts. The stored basis and M2 choices were hash-verified and were not refitted here.",
        '','## Confirmation outcomes','',
        '|Predefined outcome|n|Mean [95% bootstrap CI]|Holm p|Interpretation|','|---|---:|---:|---:|---|']
    for r in tests:
        significant=r['reject_at_05'];sign='positive' if r['mean']>0 else 'negative'
        decision=('positive signed prediction reproduced against zero' if sign=='positive' else 'opposite signed mean; positive prediction fails') if significant else 'not established against zero after correction'
        if r['status']!='tested':decision=r['status']
        lines.append(f"|{r['outcome']}|{r['n']}/{r['n_requested']}|{r['mean']:.4g} [{r['ci_lower']:.4g}, {r['ci_upper']:.4g}]|{r['p_holm']:.3g}|{decision}|")
    lines+=['','The fixed ten-test family uses approximate two-sided one-sample t tests on utterance-level signed outcomes and Holm correction. Pointwise percentile confidence intervals resample utterances within operational text-category strata. The basis is held fixed; intervals do not include model/checkpoint uncertainty or uncertainty from fitting another development cohort.',
        '','Large arrays have additional approximate simultaneous centered/standardized maximum-deviation bootstrap bands within each declared map family. These are not token-level tests. Pointwise figure ribbons must not be read as simultaneous discovery thresholds.',
        '','## Development-selected M2 hypotheses','',
        '|Hypothesis|Frozen site/layer|Development score|Confirmation mean [95% CI]|','|---|---|---:|---:|']
    for r in shortlist_rows:lines.append(f"|{r['hypothesis']}|{r['site']} / {r['layer']}|{r['development_value']:.4g}|{r['confirmation_mean']:.4g} [{r['ci_lower']:.4g}, {r['ci_upper']:.4g}]|")
    lines+=['','These are tests to consider for M2, not demonstrated control points. Their selection used development data only. No causal intervention was conducted. The frozen prediction phase is recorded in the unchanged `M2_shortlist.json`; it is not reselected from confirmation.',
        '','## What the figures measure','',
        '1. `01_shared_vs_specific.png`: transfer of the fixed shared direction and own-versus-other residualized style margin. Dashed development curves are exploratory/self-fitted; solid confirmation curves are held out.',
        '2. `02_anchor_to_generation.png`: historical speech-start references against phase-averaged generation contrasts. This cosine is distinct from the mean of pointwise cosines in `position_tertiles.csv`.',
        '3. `03_component_propagation.png`: directional decomposition of actual residual arithmetic, retaining the rounding remainder. Signed terms can exceed one or cancel; they are not causal contribution percentages.',
        '4. `04_token_predictions_by_phase.png`: final-LM full/conditional JSD by predicted code phase. No per-layer probability distribution or warped sampling probability is inferred.',
        '5. `05_text_history_robustness.png`: operational category summaries, paired final-output JSD and direction transfer between reference histories. Different histories are not word/phoneme-aligned.',
        '','## Negative findings, missingness and alternatives','']
    failed=[r['outcome'] for r in tests if not r['reject_at_05']]
    lines.append('The following predefined effects were not established after family correction: '+(', '.join(failed) if failed else 'none among the finite tested signed global outcomes')+'. A nonsignificant test does not prove equivalence or absence.')
    opposite=[r['outcome'] for r in tests if r['reject_at_05'] and r['mean']<0]
    if opposite:lines.append('Opposite-direction confirmation findings: '+', '.join(opposite)+'. These do not reproduce the proposed positive patterns.')
    missing=[r for r in data['coverage'] if r['status']!='complete']
    lines.append(f"`coverage.csv` retains {len(missing)} failed/unattempted subjects. No missing outcome is assigned zero. Identical inspected secondary histories are excluded from independent-history robustness summaries, with their duplicate status retained.")
    identical=sum(r['identical_inspected_prefix'] for r in data['history_pairs'])
    undefined=sum(r['undefined_shared_cells'] for r in data['projection_audit'])
    undefined_specific=sum(r['undefined_specificity_cells'] for r in data['projection_audit'])
    lines.append(f"Across both splits, {identical} secondary histories have identical inspected prefixes. Analysis copies mark {undefined} undefined shared-direction cells and {undefined_specific} undefined specificity cells as missing; original projected outputs remain unchanged. See `projection_missingness_audit.csv` and `paired_history_results.csv`.")
    grammar=[r for r in data['token_grammar'] if r['study_split']=='confirmation' and r['history']==0]
    lines+=['','## Secondary token-grammar diagnostic','',
        '|News-relative contrast|Full JSD (nats)|Phase-conditional JSD|News phase mass|Style phase mass|Maximum saved P(token 128258)|',
        '|---|---:|---:|---:|---:|---:|']
    for condition in LABELS:
        rows=[r for r in grammar if r['condition']==condition]
        values=[float(finite_mean([r[key] for r in rows])) for key in ('jsd_nats','phase_jsd_nats','reference_phase_mass','changed_phase_mass')]
        maxima=[r[key] for r in rows for key in ('max_reference_eos_probability','max_changed_eos_probability') if np.isfinite(r[key])]
        lines.append('|'+condition+' - News|'+''.join(f'{x:.5g}|' for x in values)+(f'{max(maxima):.5g}' if maxima else 'not available')+'|')
    exceptions=[r for r in data['token_grammar_exceptions'] if r['study_split']=='confirmation' and r['history']==0]
    lines.append(f"These descriptive means first weight phases equally within utterances. {len(exceptions)} saved contrast-position rows cross the inspection flags phase mass <0.99 or P(token 128258) >0.01; these thresholds identify diagnostics, not generated failures. The EOS diagnostic concerns token 128258 only, not the combined probability of both accepted stopping tokens (128258 and 128001). Exact rows are retained when present. If full and conditional JSD are close while phase mass stays near one, differences primarily redistribute valid-phase code probabilities. This is a conditional interpretation, not proof of emotional delivery; finite-precision residual differences and exceptions remain visible in `token_grammar_per_utterance.csv`.")
    lines+=['','Shared and style-specific lexical instruction structure, prompt position/context, residual anisotropy, reference-history policy and sampling can explain patterns without a shared emotion circuit. No lexical-invariance or emotion-delivery claim is made. A bounded future paraphrase control would be required before such a claim. The current result covers English Amit, one checkpoint, fixed generation settings, and operational lexical/referential strata. These lexical screens are not verified emotional states or context dependence: “this morning” and complementizer “that” trigger the referential screen; “love” in reviews may dominate emotion-marked examples. Category-prioritized document selection, genre, length and category remain confounded. Distinct source documents need not mean distinct authors/topics; no unseen-pretraining-data claim is supported.',
        '','Four historical alternate ANGER histories occur only in development. Confirmation history robustness compares Default seed 1234 with Default seed 5678. Their code positions are not phonetic or word alignments. Identical inspected prefixes are marked rather than reseeded/reselected to force diversity.',
        '','The first 64 complete code frames at most were inspected. This window is not guaranteed to cover an entire long recording. `per_utterance_outcomes.csv` records full-history and inspected lengths. Reference code validity is an engineering property, not a manual transcription or perceived-quality result.',
        '','## Reproducibility and source limits','',
        f"Verified {manifest['verified_available_artifacts']} locally present history artifacts. {len(manifest['omitted_tensor_artifacts'])} optional large tensor artifacts were omitted from this analysis input; see `analysis_manifest.json`. No full tensor was reconstructed from a scalar.",
        '','No new model inference, fitting of the confirmation basis, audio generation, decoder replay, training or steering is performed by this analysis script. Numerical CSVs retain contributing utterance IDs and split/category labels.',
        '','Methods: [SciPy one-sample t test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html); [Holm step-down correction, statsmodels documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).','']
    (out/'M1_RESEARCH_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true',help='Explicitly INTERIM: permit unattempted study rows')
    p.add_argument('--allow-missing-tensors',action='store_true',help='Scalar export: verify present files and report omitted optional NPZ artifacts')
    p.add_argument('--bootstrap',type=int,help='Override only for explicit engineering tests; final analysis uses frozen 2000')
    build(p.parse_args())
