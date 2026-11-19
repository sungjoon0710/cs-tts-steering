"""M1 observational summaries on fixed speech histories; no interventions.

All-position tensors/logits are temporary. Pool equally over seven predicted
SNAC phases, retain utterance identities, and keep prompt/history factors apart.
"""
from itertools import combinations
import numpy as np
import torch

from src.generation_trace import state_differences

STYLES = ('default','anger','happy','sad','news')
SITES = ('resid','attn','mlp')
CONTRASTS = [('news',x) for x in ('anger','happy','sad')]+[('default',x) for x in ('anger','happy','sad','news')]+list(combinations(('anger','happy','sad'),2))


def phase_pool(x, phases, exclude_anchor=True):
    """Pool position axis -2, keeping layer/site and hidden axes intact."""
    phases=np.asarray(phases)
    if x.shape[-2]!=len(phases) or set(phases)!={0,1,2,3,4,5,6}:
        raise ValueError('All seven aligned predicted phases are required')
    selected=np.arange(len(phases))>0 if exclude_anchor else np.ones(len(phases),bool)
    if any(not np.any((phases==p)&selected) for p in range(7)):
        raise ValueError('Every generation phase needs a non-anchor observation')
    return np.stack([np.mean(x[...,(phases==p)&selected,:],axis=-2,dtype=np.float64) for p in range(7)],axis=-2).astype(np.float32)


def cosine(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide(np.sum(a*b,axis=-1),den,out=np.full(den.shape,np.nan),where=den>0)


def fit_structure(phase_deltas):
    """Development only: N x emotion x site x layer x phase x hidden.

    One rank-1 shared consensus direction and three residualized style means.
    This is a prespecified geometric decomposition, not an emotion circuit model.
    """
    x=np.asarray(phase_deltas,np.float64)
    if x.ndim!=6 or x.shape[1]!=3 or not np.isfinite(x).all():
        raise ValueError('Expected finite development phase contrasts')
    unit=np.divide(x,np.linalg.norm(x,axis=-1,keepdims=True),out=np.zeros_like(x),where=np.linalg.norm(x,axis=-1,keepdims=True)>0)
    style=unit.mean(axis=0)
    shared=style.mean(axis=0)
    shared=np.divide(shared,np.linalg.norm(shared,axis=-1,keepdims=True),out=np.zeros_like(shared),where=np.linalg.norm(shared,axis=-1,keepdims=True)>0)
    specific=style-(style*shared).sum(axis=-1,keepdims=True)*shared
    return {'shared':shared.astype(np.float32),'style':style.astype(np.float32),'specific':specific.astype(np.float32)}


def project_structure(deltas,basis):
    x=np.asarray(deltas,np.float64);shared=np.asarray(basis['shared'],np.float64)
    unit=np.divide(x,np.linalg.norm(x,axis=-1,keepdims=True),out=np.zeros_like(x),where=np.linalg.norm(x,axis=-1,keepdims=True)>0)
    along=(unit*shared).sum(axis=-1)
    remainder=unit-along[...,None]*shared
    own=cosine(remainder,basis['specific'])
    other=np.stack([cosine(remainder,np.roll(basis['specific'],k,axis=0)) for k in (1,2)])
    count=np.isfinite(other).sum(axis=0)
    other_mean=np.divide(np.nansum(other,axis=0),count,out=np.full(count.shape,np.nan),where=count>0)
    margin=own-other_mean
    return {'shared_cosine':along.astype(np.float32),'specific_cosine':own.astype(np.float32),
            'specific_margin':margin.astype(np.float32),'shared_energy_fraction':np.square(along).astype(np.float32)}


@torch.inference_mode()
def probability_context(logits, phases, device='cpu', audio_base=128266, codebook_size=4096, eos=128258):
    lp=torch.from_numpy(np.asarray(logits)).to(device=device,dtype=torch.float32).log_softmax(-1)
    prob=lp.exp();phase_lp=torch.empty((len(phases),codebook_size),device=device)
    mass=torch.empty(len(phases),device=device)
    for phase in range(7):
        ix=torch.as_tensor(np.flatnonzero(np.asarray(phases)==phase),device=device)
        selected=lp[ix,audio_base+phase*codebook_size:audio_base+(phase+1)*codebook_size]
        total=selected.logsumexp(-1);mass[ix]=total.exp();phase_lp[ix]=selected-total[:,None]
    values,ids=lp.topk(5,dim=-1)
    return {'lp':lp,'p':prob,'conditional_lp':phase_lp,'conditional_p':phase_lp.exp(),
            'entropy':-(prob*lp).sum(-1),'phase_mass':mass,'eos_logp':lp[:,eos],
            'top_ids':ids.cpu().tolist(),'top_probabilities':values.exp().cpu().tolist()}


@torch.inference_mode()
def probability_contrast(a,b,targets,phases):
    logm=torch.logaddexp(a['lp'],b['lp'])-np.log(2)
    full=(.5*((a['p']*(a['lp']-logm)).sum(-1)+(b['p']*(b['lp']-logm)).sum(-1))).clamp_min(0)
    logc=torch.logaddexp(a['conditional_lp'],b['conditional_lp'])-np.log(2)
    conditional=(.5*((a['conditional_p']*(a['conditional_lp']-logc)).sum(-1)+(b['conditional_p']*(b['conditional_lp']-logc)).sum(-1))).clamp_min(0)
    ix=torch.arange(len(targets),device=a['lp'].device);target=torch.tensor(targets,device=ix.device)
    values={'jsd_nats':full,'phase_jsd_nats':conditional,'reference_target_logp':a['lp'][ix,target],
        'changed_target_logp':b['lp'][ix,target], 'delta_target_logp':b['lp'][ix,target]-a['lp'][ix,target],
        'reference_phase_mass':a['phase_mass'],'changed_phase_mass':b['phase_mass'],
        'reference_eos_probability':a['eos_logp'].exp(),'changed_eos_probability':b['eos_logp'].exp(),
        'reference_eos_logp':a['eos_logp'],'changed_eos_logp':b['eos_logp'],
        'reference_entropy_nats':a['entropy'],'changed_entropy_nats':b['entropy'],
        'argmax_changed':(a['lp'].argmax(-1)!=b['lp'].argmax(-1)).float()}
    values={k:v.cpu().numpy() for k,v in values.items()}
    return [{'j':j,'predicted_phase':int(phases[j]),'target_token':int(targets[j]),
             **{k:float(v[j]) for k,v in values.items()}} for j in range(len(targets))]


def representation_contrast(a,b,phases):
    maps,rows={},[]
    for site in SITES:
        cd,l2=state_differences(a['states'][site],b['states'][site])
        maps[site+'_cosine_distance']=cd.astype(np.float32)
        maps[site+'_normalized_l2']=l2.astype(np.float32)
        for l in range(cd.shape[0]):
            for p in range(7):
                mask=np.asarray(phases)==p
                rows.append({'site':site,'layer':l,'phase':p,'n_positions':int(mask.sum()),
                    'cosine_distance':float(np.mean(cd[l,mask])),'normalized_l2':float(np.mean(l2[l,mask]))})
    return maps,rows


def residual_decomposition(a,b,phases):
    """Directional projections along Δ post-block residual, NOT causal fractions.

    Include the actual rounding remainder. Inputs at the replay positions are
    identical token embeddings, so layer-0 incoming difference is exactly zero.
    """
    dr=b['states']['resid'].astype(np.float64)-a['states']['resid']
    da=b['states']['attn'].astype(np.float64)-a['states']['attn']
    dm=b['states']['mlp'].astype(np.float64)-a['states']['mlp']
    incoming=np.concatenate([np.zeros_like(dr[:1]),dr[:-1]],axis=0)
    rounding=dr-incoming-da-dm;den=(dr*dr).sum(-1)
    parts={key:np.divide((v*dr).sum(-1),den,out=np.full(den.shape,np.nan),where=den>1e-20)
           for key,v in [('incoming',incoming),('attention',da),('mlp',dm),('rounding',rounding)]}
    rows=[]
    for l in range(dr.shape[0]):
        for p in range(7):
            mask=(np.asarray(phases)==p)&(np.arange(len(phases))>0)
            rows.append({'layer':l,'phase':p,**{k:float(np.mean(v[l,mask][np.isfinite(v[l,mask])])) if np.isfinite(v[l,mask]).any() else np.nan for k,v in parts.items()},
                         'n_nonzero':int(np.sum(den[l,mask]>1e-20))})
    return rows


def holm(pvalues):
    p=np.asarray(pvalues,np.float64)
    if np.any((p<0)|(p>1)|~np.isfinite(p)):
        raise ValueError('Finite p-values in [0,1] required')
    order=np.argsort(p);adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order])
    result=np.empty(len(p));result[order]=np.minimum(adjusted,1)
    return result
